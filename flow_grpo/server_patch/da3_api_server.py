#!/usr/bin/env python
"""
API server for pose estimation evaluation.
Run with: python da3_api_server.py --port 34567

conda create -n da3 --clone PyTorch-2.1.0
conda activate da3
pip install moviepy==1.0.3 plyfile  trimesh evo e3nn fastapi uvicorn
cd /home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/flow_grpo/server_patch
ASCEND_RT_VISIBLE_DEVICES=7 python da3_api_server.py
"""
import argparse
import json
import os
import re
import cv2
import numpy as np
import torch
import torch_npu
import tempfile
import time
import uuid
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from depth_anything_3.api import DepthAnything3
import uvicorn
from threading import Lock

app = FastAPI(title="Pose Estimation Evaluation API")

# Add global exception handler to catch all unhandled exceptions
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to ensure all errors are logged."""
    import traceback
    error_detail = f"Unhandled exception: {str(exc)}\nTraceback:\n{traceback.format_exc()}"
    print(f"[DA3 API Server] Global exception handler: {error_detail}")
    import sys
    sys.stderr.write(f"[DA3 API Server] Global exception details:\n{error_detail}\n")
    sys.stderr.flush()
    return JSONResponse(
        status_code=500,
        content={"detail": error_detail}
    )

# Global model instance and lock for thread-safe loading
_model: Optional[DepthAnything3] = None
_model_lock = Lock()
_model_dir: Optional[str] = None
_device: Optional[torch.device] = None

class EvaluationRequest(BaseModel):
    video_path: str
    gt_json_data: Dict[str, str]  # Simplified format: {"frame0": "[...]", "frame1": "[...]", ...}

class EvaluationResponse(BaseModel):
    rot_err_mean: float
    trans_err_mean: float


def extract_frames_from_video(video_path, output_dir, fps=1.0):
    """Extract frames from video file."""
    import traceback
    from PIL import Image
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Cannot open video: {video_path}"
            print(f"[extract_frames_from_video] {error_msg}")
            raise ValueError(error_msg)
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps)) if video_fps > 0 else 1
        
        frames_dir = os.path.join(output_dir, "input_images")
        os.makedirs(frames_dir, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        frame_paths = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(frames_dir, f"{saved_count:06d}.png")
                
                # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use PIL to save image, which is more reliable
                pil_img = Image.fromarray(frame_rgb)
                pil_img.save(frame_path, format='PNG', quality=100)
                
                # Verify file was written correctly
                if not os.path.exists(frame_path):
                    raise IOError(f"Failed to save frame to {frame_path}")
                
                # Verify file is readable
                try:
                    test_img = Image.open(frame_path)
                    test_img.verify()  # Verify it's a valid image
                    test_img.close()
                except Exception as verify_err:
                    raise IOError(f"Saved frame {frame_path} is corrupted: {verify_err}")
                
                frame_paths.append(frame_path)
                saved_count += 1
            frame_count += 1
        
        cap.release()
        print(f"[extract_frames_from_video] Extracted {len(frame_paths)} frames from {video_path}")
        print(f"[extract_frames_from_video] Frames saved to: {frames_dir}")
        return sorted(frame_paths)
    except Exception as e:
        error_detail = f"Error extracting frames from {video_path}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[extract_frames_from_video] {error_detail}")
        import sys
        sys.stderr.write(f"[extract_frames_from_video] Full error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise


def parse_matrix_string(matrix_str):
    """Parse matrix string into 4x4 numpy array."""
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    numbers = [float(n) for n in re.findall(pattern, matrix_str)]
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers, got {len(numbers)}")
    return np.array(numbers).reshape(4, 4)


def load_gt_extrinsics_from_dict(gt_json_data: Dict[str, str]):
    """
    Load ground truth extrinsics from simplified JSON data.
    
    Args:
        gt_json_data: Dictionary with format {"frame0": "[...]", "frame1": "[...]", ...}
    
    Returns:
        numpy array of shape (N, 3, 4) containing extrinsics matrices
    """
    import traceback
    try:
        frames = sorted([k for k in gt_json_data.keys() if k.startswith("frame")], 
                        key=lambda x: int(x.replace("frame", "")))
        
        if not frames:
            raise ValueError("No frame data found in gt_json_data")
        
        extrinsics_list = []
        for frame_key in frames:
            matrix_str = gt_json_data[frame_key]
            matrix_4x4 = parse_matrix_string(matrix_str)
            extrinsics_list.append(matrix_4x4[:3, :])  # Extract 3x4 part
        
        result = np.array(extrinsics_list)
        print(f"[load_gt_extrinsics_from_dict] Loaded {len(result)} extrinsics matrices")
        return result
    except Exception as e:
        error_detail = f"Error loading GT extrinsics: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[load_gt_extrinsics_from_dict] {error_detail}")
        import sys
        sys.stderr.write(f"[load_gt_extrinsics_from_dict] Full error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise


def compute_rotation_error(R_gt, R_pred):
    """Compute rotation error (RotErr) in radians."""
    R_rel = R_gt @ R_pred.T
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_translation_error(t_gt, t_pred):
    """Compute translation error (TransErr)."""
    return np.linalg.norm(t_gt - t_pred)


def load_model(model_dir: str = "/home/ma-user/modelarts/user-job-dir/wlh/Model/depth-anything/DA3-GIANT"):
    """
    Load model to GPU memory (thread-safe, only loads once).
    
    Args:
        model_dir: Model directory path
    
    Returns:
        tuple: (model, device)
    """
    import traceback
    global _model, _model_dir, _device
    
    with _model_lock:
        if _model is None:
            try:
                print(f"[DA3 API Server] Loading model from {model_dir}...")
                _device = torch.device("npu" if torch.npu.is_available() else "cpu")
                print(f"[DA3 API Server] Using device: {_device}")
                _model = DepthAnything3.from_pretrained(model_dir)
                print(f"[DA3 API Server] Model created, moving to device...")
                _model = _model.to(device=_device)
                _model_dir = model_dir
                print(f"[DA3 API Server] Model loaded on {_device}")
            except Exception as e:
                error_detail = f"Error loading model from {model_dir}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                print(f"[DA3 API Server] {error_detail}")
                import sys
                sys.stderr.write(f"[DA3 API Server] Load model error details:\n{error_detail}\n")
                sys.stderr.flush()
                raise
        else:
            print(f"[DA3 API Server] Using cached model on {_device}")
    
    return _model, _device


def unload_model():
    """
    Unload model from GPU memory and free resources.
    """
    global _model, _model_dir, _device
    
    with _model_lock:
        if _model is not None:
            device_str = str(_device) if _device is not None else "unknown"
            
            # Print memory status before unloading
            print(f"[DA3 API Server] Memory status before unload:")
            if torch.npu.is_available():
                print(f"[DA3 API Server] NPU memory allocated: {torch.npu.memory_allocated() / 1024**2:.2f} MB")
                print(f"[DA3 API Server] NPU memory reserved: {torch.npu.memory_reserved() / 1024**2:.2f} MB")
            elif torch.cuda.is_available():
                print(f"[DA3 API Server] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"[DA3 API Server] CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            print(f"[DA3 API Server] Unloading model from {device_str}...")
            
            # Move model to CPU before deletion to ensure proper cleanup
            try:
                if hasattr(_model, 'to'):
                    _model = _model.to('cpu')
                # Also try to clear any internal caches or buffers
                if hasattr(_model, 'eval'):
                    _model.eval()  # Set to eval mode to disable any training-specific buffers
                # Try to clear any cached attributes
                if hasattr(_model, 'input_processor'):
                    try:
                        del _model.input_processor
                    except:
                        pass
            except Exception as e:
                print(f"[DA3 API Server] Warning: Error moving model to CPU: {e}")
            
            # Delete model and force garbage collection
            del _model
            _model = None
            _model_dir = None
            _device = None
            
            # Force Python garbage collection multiple times
            import gc
            for _ in range(3):
                gc.collect()
            
            # Clear GPU cache multiple times to ensure all memory is freed
            if torch.npu.is_available():
                for _ in range(3):
                    torch.npu.empty_cache()
                torch.npu.synchronize()
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                for _ in range(3):
                    torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            print(f"[DA3 API Server] Model unloaded and memory freed from {device_str}")
            print(f"[DA3 API Server] Memory status after unload:")
            if torch.npu.is_available():
                allocated = torch.npu.memory_allocated() / 1024**2
                reserved = torch.npu.memory_reserved() / 1024**2
                print(f"[DA3 API Server] NPU memory allocated: {allocated:.2f} MB")
                print(f"[DA3 API Server] NPU memory reserved: {reserved:.2f} MB")
                if allocated > 100:  # If more than 100MB is still allocated, warn
                    print(f"[DA3 API Server] WARNING: {allocated:.2f} MB still allocated after unload!")
            elif torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"[DA3 API Server] CUDA memory allocated: {allocated:.2f} MB")
                print(f"[DA3 API Server] CUDA memory reserved: {reserved:.2f} MB")
                if allocated > 100:  # If more than 100MB is still allocated, warn
                    print(f"[DA3 API Server] WARNING: {allocated:.2f} MB still allocated after unload!")
        else:
            print(f"[DA3 API Server] Model is already unloaded")


def evaluate_pose_estimation(
    video_path: str,
    gt_json_data: Dict[str, str],
    model_dir: str = "/home/ma-user/modelarts/user-job-dir/wlh/Model/depth-anything/DA3-GIANT",
    fps: float = 15.0,
    export_format: str = "glb",
    auto_load: bool = True
):
    """
    Evaluate camera pose estimation accuracy.
    Uses globally cached model to avoid reloading.
    
    Args:
        video_path: Path to input video file
        gt_json_data: Dictionary with format {"frame0": "[...]", "frame1": "[...]", ...}
        model_dir: Model directory path (only used on first call)
        fps: Sampling FPS for frame extraction
        export_format: Export format for model inference
        auto_load: If True, automatically load model if not loaded (default: True for backward compatibility)
    
    Returns:
        tuple: (RotErr_Mean in radians, TransErr_Mean)
    """
    import traceback
    
    try:
        # Load model (will use cached version if already loaded)
        # If auto_load=False and model is not loaded, raise an error
        if not auto_load and _model is None:
            raise RuntimeError("Model is not loaded. Please call /load_model endpoint first.")
        
        if _model is None:
            print(f"[evaluate_pose_estimation] Auto-loading model from {model_dir}...")
            model, device = load_model(model_dir)
            print(f"[evaluate_pose_estimation] Model auto-loaded on {device}")
        else:
            model, device = _model, _device
            print(f"[evaluate_pose_estimation] Using cached model on {device}")
        
        # Create unique output directory for this request to avoid concurrent access conflicts
        request_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        output_dir = os.path.join(tempfile.gettempdir(), f"da3_eval_{request_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Extract frames and run inference
            print(f"[evaluate_pose_estimation] Extracting frames from {video_path}...")
            print(f"[evaluate_pose_estimation] Using output directory: {output_dir}")
            images = extract_frames_from_video(video_path, output_dir=output_dir, fps=fps)
            print(f"[evaluate_pose_estimation] Extracted {len(images)} frames")
            
            print(f"[evaluate_pose_estimation] Running inference (extrinsics only, no export)...")
            # Only get extrinsics, skip file export and other unnecessary computations
            prediction = model.inference(
                images, 
                export_dir=None,  # Don't export any files
                export_format=None,  # No export format needed
                extrinsics_only=True,  # Optimization: skip unnecessary computations
            )
            print(f"[evaluate_pose_estimation] Inference completed")
            
            # Clean up images list immediately after inference
            del images
            import gc
            gc.collect()
            
            # Load GT and compute errors
            print(f"[evaluate_pose_estimation] Loading GT extrinsics...")
            gt_extrinsics = load_gt_extrinsics_from_dict(gt_json_data)
            print(f"[evaluate_pose_estimation] GT extrinsics shape: {gt_extrinsics.shape}")
            
            pred_extrinsics = prediction.extrinsics
            print(f"[evaluate_pose_estimation] Predicted extrinsics shape: {pred_extrinsics.shape}")
            
            N = min(len(gt_extrinsics), len(pred_extrinsics))
            print(f"[evaluate_pose_estimation] Computing errors for {N} frames...")
            rot_errors, trans_errors = [], []
            
            for i in range(N):
                R_gt, R_pred = gt_extrinsics[i, :3, :3], pred_extrinsics[i, :3, :3]
                t_gt, t_pred = gt_extrinsics[i, :3, 3], pred_extrinsics[i, :3, 3]
                rot_errors.append(compute_rotation_error(R_gt, R_pred))
                trans_errors.append(compute_translation_error(t_gt, t_pred))
            
            rot_err_mean = np.mean(rot_errors)
            trans_err_mean = np.mean(trans_errors)
            print(f"[evaluate_pose_estimation] Computed errors: rot_err_mean={rot_err_mean:.6f}, trans_err_mean={trans_err_mean:.6f}")
            
            # Clean up intermediate variables before returning
            del prediction, gt_extrinsics, pred_extrinsics, rot_errors, trans_errors
            gc.collect()
            
            # Clear GPU/NPU cache after computation
            torch.npu.empty_cache()
            torch.npu.synchronize()
            
            return rot_err_mean, trans_err_mean
        finally:
            # Clean up temporary directory after inference
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                    print(f"[evaluate_pose_estimation] Cleaned up temporary directory: {output_dir}")
            except Exception as cleanup_err:
                print(f"[evaluate_pose_estimation] Warning: Failed to cleanup {output_dir}: {cleanup_err}")
    
    except Exception as e:
        error_detail = f"Error in evaluate_pose_estimation: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[evaluate_pose_estimation] {error_detail}")
        import sys
        sys.stderr.write(f"[evaluate_pose_estimation] Full error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise


@app.post('/evaluate_pose', response_model=EvaluationResponse)
async def evaluate_pose(request: EvaluationRequest):
    """
    Evaluate camera pose estimation accuracy.
    Model will be auto-loaded on first request if not already loaded.
    For explicit control, use /load_model and /unload_model endpoints.
    
    Request body (JSON):
    {
        "video_path": "path/to/video.mp4",
        "gt_json_data": {
            "frame0": "[...]",
            "frame1": "[...]",
            ...
            "frame80": "[...]"
        }
    }
    
    Response (JSON):
    {
        "rot_err_mean": 0.001234,
        "trans_err_mean": 0.056789
    }
    """
    if not request.gt_json_data:
        raise HTTPException(status_code=400, detail="gt_json_data is empty")
    
    try:
        # Call evaluation function directly with JSON data
        # auto_load=True: automatically load model if not loaded (backward compatibility)
        rot_err_mean, trans_err_mean = evaluate_pose_estimation(
            video_path=request.video_path,
            gt_json_data=request.gt_json_data,
            auto_load=True
        )
        
        # Return results
        return EvaluationResponse(
            rot_err_mean=float(rot_err_mean),
            trans_err_mean=float(trans_err_mean)
        )
        
    except FileNotFoundError as e:
        import traceback
        error_detail = f"File not found: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[DA3 API Server] FileNotFoundError: {error_detail}")
        raise HTTPException(status_code=404, detail=error_detail)
    except ValueError as e:
        import traceback
        error_detail = f"Value error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[DA3 API Server] ValueError: {error_detail}")
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        import traceback
        error_detail = f"Internal server error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[DA3 API Server] Exception: {error_detail}")
        import sys
        sys.stderr.write(f"[DA3 API Server] Full error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=error_detail)


@app.get('/health')
async def health():
    """Health check endpoint."""
    global _model
    model_status = "loaded" if _model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": str(_device) if _device is not None else "unknown"
    }


@app.post('/unload_model')
async def unload_model_endpoint():
    """
    Unload model from GPU memory.
    Call this after all video evaluations are complete.
    """
    try:
        unload_model()
        return {"status": "success", "message": "Model unloaded from GPU memory"}
    except Exception as e:
        import traceback
        error_detail = f"Error unloading model: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[DA3 API Server] Unload model error: {error_detail}")
        import sys
        sys.stderr.write(f"[DA3 API Server] Unload model error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=error_detail)


@app.post('/load_model')
async def load_model_endpoint(model_dir: Optional[str] = None):
    """
    Explicitly load model to GPU memory.
    Usually not needed as model loads automatically on first request.
    """
    try:
        default_model_dir = "/home/ma-user/modelarts/user-job-dir/wlh/Model/depth-anything/DA3-GIANT"
        model_dir = model_dir or default_model_dir
        model, device = load_model(model_dir)
        return {
            "status": "success",
            "message": f"Model loaded on {device}",
            "device": str(device)
        }
    except Exception as e:
        import traceback
        error_detail = f"Error loading model: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[DA3 API Server] Load model error: {error_detail}")
        import sys
        sys.stderr.write(f"[DA3 API Server] Load model error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=error_detail)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Estimation Evaluation API Server')
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on (default: 34567 + RANK)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--base-port', type=int, default=34567, help='Base port number (default: 34567, actual port = base_port + RANK)')
    
    args = parser.parse_args()
    
    # Get RANK from environment variable (for multi-GPU setup)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    
    # Calculate port: use specified port, or base_port + rank
    if args.port is not None:
        port = args.port
    else:
        port = args.base_port + rank
    
    print(f"[RANK {rank}] Starting API server on {args.host}:{port}")
    print(f"[RANK {rank}] Health check: http://{args.host}:{port}/health")
    print(f"[RANK {rank}] API endpoint: http://{args.host}:{port}/evaluate_pose")
    print(f"[RANK {rank}] API docs: http://{args.host}:{port}/docs")
    
    uvicorn.run(app, host=args.host, port=port)
