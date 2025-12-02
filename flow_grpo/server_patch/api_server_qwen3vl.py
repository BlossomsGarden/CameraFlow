#!/usr/bin/env python
"""
API server for UnifiedReward-2.0-qwen3vl-8b video evaluation.
Run with: python api_server_qwen3vl.py --port 34569


conda create -n unifiedrewardqwen3 --clone qwen3vl
conda activate unifiedrewardqwen3
pip install fastapi uvicorn
ASCEND_RT_VISIBLE_DEVICES="0" python api_server_qwen3vl.py
"""
import argparse
import os
import warnings
import torch
import torch_npu
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Optional
import uvicorn
from threading import Lock
import traceback
import gc

warnings.filterwarnings("ignore")

app = FastAPI(title="UnifiedReward-2.0-qwen3vl-8b Video Evaluation API")

# Add global exception handler to catch all unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler to ensure all errors are logged."""
    error_detail = f"Unhandled exception: {str(exc)}\nTraceback:\n{traceback.format_exc()}"
    print(f"[Qwen3VL API Server] Global exception handler: {error_detail}")
    import sys
    sys.stderr.write(f"[Qwen3VL API Server] Global exception details:\n{error_detail}\n")
    sys.stderr.flush()
    return JSONResponse(
        status_code=500,
        content={"detail": error_detail}
    )

# Global model and processor with thread-safe loading
_model: Optional[Qwen3VLForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None
_model_lock = Lock()
_model_path: Optional[str] = None
_device: Optional[torch.device] = None


class EvaluationRequest(BaseModel):
    video_path: str
    prompt: str
    fps: float = 30.0  # 默认 fps

class EvaluationResponse(BaseModel):
    output_text: str

def get_evaluation_prompt(video_prompt: str) -> str:
    """
    生成评估模型的提示词
    
    Args:
        video_prompt: 视频生成的文本提示词
    
    Returns:
        完整的评估提示词
    """
    model_prompt = f'''You are an objective and precise evaluator for video quality assessment. I will provide you with a text caption and a generated video based on that caption. Evaluate the video based on the caption: 「{video_prompt}」of four dimensions I provid and then give me the average score. 
        The video is from a UE5 virtual scene (synthetic but realistic-looking), so focus on content quality and consistency, not realism or authenticity.

        TOTAL FOUR EVALUATION DIMENSIONS (You MUST evaluate the video across these four core dimensions STRICTLY):
        Dimension 1: Frame-to-Frame Consistency:
           - Check if objects, characters, and scene elements maintain consistent shapes, colors, textures, and positions across frames
           - Assess whether objects remain stable when camera viewpoint changes
           - Look for flickering, jittering, or sudden appearance/disappearance of objects
           - Evaluate if character appearances (facial features, body proportions, clothing) remain consistent throughout the video
        
        Dimension 2: Visual Quality and Artifacts:
           - Check for black borders, edge artifacts, or incomplete frames
           - Identify if there are any visual artifacts or distortions
           - Assess if characters' body shapes is properly rendered without deformation or collapse
           - Check if objects maintain their proper shapes without distortion or morphing"

        Dimension 3: Prompt Alignment:
           - Assess how well the video content matches the provided text prompt
           - Check if described objects, characters, actions, and scene elements are present and correctly depicted
           - Evaluate if the relationships between elements match the prompt description
           - Verify that attributes (colors, clothing, poses, etc.) align with the prompt"
        
        Dimension 4: Scene Coherence:
           - Evaluate the smoothness and naturalness of character movements and animations
           - Check if the scene maintains logical spatial relationships throughout the video
           - Assess if lighting and shadows remain consistent across frames
           - Evaluate if background elements remain stable and coherent"


        For each of the four evaluation dimensions mentioned above, get Score1, Score2, Score3, Score4 respectively:
            - Score1, Score2, Score3, Score4 (in a format "X.XX") must be between 1.00 and 5.00 with two decimal places (e.g., 3.51, 4.20, 2.85). NOT 0-1, NOT 0-10, NOT any other scale!
            - Lower scores (1.00-3.00) for videos with significant issues (artifacts, inconsistencies, poor alignment). Higher scores (3.00-5.00) for high-quality, consistent, and well-aligned videos
            - Provide a BRIEF rationale for the score: ONLY 1-2 SHORT sentences. Only one line. Do NOT use numbered lists (#1, #2, #3). Do NOT write long paragraphs. Keep it concise!
            - Each of the four dimension must follow exactly this format:
                Dimension 1: Frame-to-Frame Consistency: 
                    Score1: X.XX - A Brief rationale about its flaw...
                Dimension 2: Visual Quality and Artifacts: 
                    Score2: X.XX - A Brief rationale about its flaw...
                Dimension 3: Prompt Alignment: 
                    Score3: X.XX - A Brief rationale about its flaw...
                Dimension 4: Scene Coherence: 
                    Score4: X.XX - A Brief rationale about its flaw...

        
        After evaluating all FOUR dimensions mentioned above, then calculate the Average score as the final score:
        - Take the four scores Score1, Score2, Score3, Score4 from Dimensions 1, 2, 3, and 4, then calculate: (Score1 + Score2 + Score3 + Score4) / 4 = X.XX
        - The average is calculated from the FOUR main dimension scores Score1, Score2, Score3, Score4, NOT from sub-items within each dimension or other scores.
        - The calculated average (X.XX) must be shown with EXACTLY TWO decimal places (e.g., 3.45, 4.20, 2.75, 3.48). NOT a long decimal like 1.3303339920043945 - round to TWO decimal places!
        - **CRITICAL: You MUST show the calculation explicitly, following this EXACT format (write it ONCE, do NOT repeat):**
            Average score:
            (X.XX + X.XX + X.XX + X.XX) / 4 = X.XX


        Instructions (MUST FOLLOW STRICTLY):
        1. After evaluating all four dimensions and calculating the average score, you MUST ALWAYS output the final average score inside <answer> and </answer> tags. The <answer></answer> tag is MANDATORY - you CANNOT skip it or omit it.
        2. **CRITICAL: The number inside <answer></answer> MUST be EXACTLY the average of Score1, Score2, Score3, Score4 that you calculated. It MUST be (Score1 + Score2 + Score3 + Score4) / 4. You CANNOT use any other number or arbitrary value. The <answer> value MUST match the calculated average from the formula.**
        3. The <answer> tag MUST contain ONLY the number with TWO decimal places like "X.XX" (e.g. 3.45, 4.20). NO other text, NO calculations, NO equations, NO "average:", NO other words in <answer>. ONLY the number with TWO decimal places.
        

        Required OUTPUT FORMAT Example (MUST FOLLOW STRICTLY, including line breaks and indentation):

        Dimension 1: Frame-to-Frame Consistency: 
            Score1: 3.30 - Brief rationale about its flaw... (1-2 sentences)
        Dimension 2: Visual Quality and Artifacts: 
            Score2: 3.33 - Brief rationale about its flaw... (1-2 sentences)
        Dimension 3: Prompt Alignment: 
            Score3: 4.12 - Brief rationale about its flaw... (1-2 sentences)
        Dimension 4: Scene Coherence: 
            Score4: 2.85 - Brief rationale about its flaw... (1-2 sentences)
        Average score:
        (3.30 + 3.33 + 4.12 + 2.85) / 4 = 3.40

        <answer>3.40</answer>

        Note: The example numbers above are only to illustrate the exact format. Your actual evaluation must follow this format exactly STRICTLY.
        '''
    return model_prompt


def load_model(model_dir: str = "/home/ma-user/modelarts/user-job-dir/wlh/Model/CodeGoat24/UnifiedReward-2.0-qwen3vl-8b"):
    """
    Load model to GPU memory (thread-safe, only loads once).
    
    Args:
        model_dir: Model directory path
    
    Returns:
        tuple: (model, processor, device)
    """
    global _model, _processor, _model_path, _device
    
    with _model_lock:
        if _model is None:
            try:
                print(f"[Qwen3VL API Server] Loading model from {model_dir}...")
                _device = torch.device("npu" if torch.npu.is_available() else "cpu")
                print(f"[Qwen3VL API Server] Using device: {_device}")
                _model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                _processor = AutoProcessor.from_pretrained(model_dir)
                _model_path = model_dir
                print(f"[Qwen3VL API Server] Model loaded on {_device}")
            except Exception as e:
                error_detail = f"Error loading model from {model_dir}: {str(e)}\nTraceback:\n{traceback.format_exc()}"
                print(f"[Qwen3VL API Server] {error_detail}")
                import sys
                sys.stderr.write(f"[Qwen3VL API Server] Load model error details:\n{error_detail}\n")
                sys.stderr.flush()
                raise
        else:
            print(f"[Qwen3VL API Server] Using cached model on {_device}")
    
    return _model, _processor, _device


def unload_model():
    """
    Unload model from GPU memory and free resources.
    """
    global _model, _processor, _model_path, _device
    
    with _model_lock:
        if _model is not None:
            device_str = str(_device) if _device is not None else "unknown"
            
            # Print memory status before unloading
            print(f"[Qwen3VL API Server] Memory status before unload:")
            if torch.npu.is_available():
                print(f"[Qwen3VL API Server] NPU memory allocated: {torch.npu.memory_allocated() / 1024**2:.2f} MB")
                print(f"[Qwen3VL API Server] NPU memory reserved: {torch.npu.memory_reserved() / 1024**2:.2f} MB")
            elif torch.cuda.is_available():
                print(f"[Qwen3VL API Server] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"[Qwen3VL API Server] CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            print(f"[Qwen3VL API Server] Unloading model from {device_str}...")
            
            # Move model to CPU before deletion to ensure proper cleanup
            try:
                if hasattr(_model, 'to'):
                    _model = _model.to('cpu')
                if hasattr(_model, 'eval'):
                    _model.eval()  # Set to eval mode to disable any training-specific buffers
            except Exception as e:
                print(f"[Qwen3VL API Server] Warning: Error moving model to CPU: {e}")
            
            # Delete model and processor
            del _model
            del _processor
            _model = None
            _processor = None
            _model_path = None
            _device = None
            
            # Force Python garbage collection multiple times
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
            
            print(f"[Qwen3VL API Server] Model unloaded and memory freed from {device_str}")
            print(f"[Qwen3VL API Server] Memory status after unload:")
            if torch.npu.is_available():
                allocated = torch.npu.memory_allocated() / 1024**2
                reserved = torch.npu.memory_reserved() / 1024**2
                print(f"[Qwen3VL API Server] NPU memory allocated: {allocated:.2f} MB")
                print(f"[Qwen3VL API Server] NPU memory reserved: {reserved:.2f} MB")
                if allocated > 100:  # If more than 100MB is still allocated, warn
                    print(f"[Qwen3VL API Server] WARNING: {allocated:.2f} MB still allocated after unload!")
            elif torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"[Qwen3VL API Server] CUDA memory allocated: {allocated:.2f} MB")
                print(f"[Qwen3VL API Server] CUDA memory reserved: {reserved:.2f} MB")
                if allocated > 100:  # If more than 100MB is still allocated, warn
                    print(f"[Qwen3VL API Server] WARNING: {allocated:.2f} MB still allocated after unload!")
        else:
            print(f"[Qwen3VL API Server] Model is already unloaded")


def evaluate_video_quality(video_path: str, prompt: str, fps: float = 30.0, auto_load: bool = True) -> str:
    """
    Evaluate video quality and alignment with text caption using UnifiedReward-2.0-qwen3vl-8b model.
    Uses globally cached model to avoid reloading.
    
    Args:
        video_path: Path to the video file
        prompt: Text caption describing the video content
        fps: Frame rate for video processing (default: 30.0)
        auto_load: If True, automatically load model if not loaded (default: True for backward compatibility)
    
    Returns:
        Evaluation output text containing the score
    """
    global _model, _processor
    
    # Load model (will use cached version if already loaded)
    # If auto_load=False and model is not loaded, raise an error
    if not auto_load and _model is None:
        raise RuntimeError("Model is not loaded. Please call /load_model endpoint first.")
    
    if _model is None:
        print(f"[evaluate_video_quality] Auto-loading model...")
        model, processor, device = load_model()
        print(f"[evaluate_video_quality] Model auto-loaded on {device}")
    else:
        model, processor = _model, _processor
        print(f"[evaluate_video_quality] Using cached model")
    
    # 生成评估提示词
    model_prompt = get_evaluation_prompt(prompt)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps
                },
                {"type": "text", "text": model_prompt},
            ],
        }
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    generated_ids = None
    generated_ids_trimmed = None
    output_text = None
    
    try:
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
    finally:
        # 清理显存：删除所有中间变量，确保显存完全释放
        # 参考 da3_api_server.py 的清理方式
        try:
            # Move tensors to CPU before deletion to free GPU memory
            if inputs is not None:
                if isinstance(inputs, dict):
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.cpu()
                            del value
                del inputs
        except Exception as e:
            print(f"[evaluate_video_quality] Warning: Error cleaning up inputs: {e}")
        
        try:
            if generated_ids is not None:
                if isinstance(generated_ids, torch.Tensor):
                    generated_ids = generated_ids.cpu()
                del generated_ids
        except Exception as e:
            print(f"[evaluate_video_quality] Warning: Error cleaning up generated_ids: {e}")
        
        try:
            if generated_ids_trimmed is not None:
                if isinstance(generated_ids_trimmed, list) and len(generated_ids_trimmed) > 0:
                    if isinstance(generated_ids_trimmed[0], torch.Tensor):
                        for t in generated_ids_trimmed:
                            if isinstance(t, torch.Tensor):
                                t_cpu = t.cpu()
                                del t
                        generated_ids_trimmed = []
                del generated_ids_trimmed
        except Exception as e:
            print(f"[evaluate_video_quality] Warning: Error cleaning up generated_ids_trimmed: {e}")
        
        # Clean up other variables
        try:
            del model_prompt, messages
        except Exception:
            pass
        
        # Force Python garbage collection multiple times (like da3_api_server.py)
        gc.collect()
        
        torch.npu.empty_cache()
        torch.npu.synchronize()
        torch.npu.empty_cache()
    
    return output_text

# Removed @app.on_event("startup") - model will be loaded on first request or via /load_model endpoint


@app.post('/evaluate_video', response_model=EvaluationResponse)
async def evaluate_video(request: EvaluationRequest):
    """
    Evaluate video quality and alignment with text caption.
    Model will be auto-loaded on first request if not already loaded.
    For explicit control, use /load_model and /unload_model endpoints.
    
    Request body (JSON):
    {
        "video_path": "path/to/video.mp4",
        "prompt": "A person stands near a train platform...",
        "fps": 30.0
    }
    
    Response (JSON):
    {
        "output_text": "Final Score: 4.5"
    }
    """
    try:
        # Call evaluation function directly
        # auto_load=True: automatically load model if not loaded (backward compatibility)
        output_text = evaluate_video_quality(
            video_path=request.video_path,
            prompt=request.prompt,
            fps=request.fps,
            auto_load=True
        )
        
        # Return results
        return EvaluationResponse(
            output_text=output_text
        )
        
    except FileNotFoundError as e:
        import traceback
        error_detail = f"File not found: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[Qwen3VL API Server] FileNotFoundError: {error_detail}")
        raise HTTPException(status_code=404, detail=error_detail)
    except ValueError as e:
        import traceback
        error_detail = f"Value error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[Qwen3VL API Server] ValueError: {error_detail}")
        raise HTTPException(status_code=400, detail=error_detail)
    except RuntimeError as e:
        import traceback
        error_detail = f"Model not available: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[Qwen3VL API Server] RuntimeError: {error_detail}")
        raise HTTPException(status_code=503, detail=error_detail)
    except Exception as e:
        import traceback
        error_detail = f"Internal server error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[Qwen3VL API Server] Exception: {error_detail}")
        import sys
        sys.stderr.write(f"[Qwen3VL API Server] Full error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=error_detail)


@app.get('/health')
async def health():
    """Health check endpoint."""
    global _model, _processor, _device
    model_status = "loaded" if (_model is not None and _processor is not None) else "not_loaded"
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
        print(f"[Qwen3VL API Server] Unload model error: {error_detail}")
        import sys
        sys.stderr.write(f"[Qwen3VL API Server] Unload model error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=error_detail)


@app.post('/load_model')
async def load_model_endpoint(model_dir: Optional[str] = None):
    """
    Explicitly load model to GPU memory.
    Usually not needed as model loads automatically on first request.
    """
    try:
        default_model_dir = "/home/ma-user/modelarts/user-job-dir/wlh/Model/CodeGoat24/UnifiedReward-2.0-qwen3vl-8b"
        model_dir = model_dir or default_model_dir
        model, processor, device = load_model(model_dir)
        return {
            "status": "success",
            "message": f"Model loaded on {device}",
            "device": str(device)
        }
    except Exception as e:
        import traceback
        error_detail = f"Error loading model: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        print(f"[Qwen3VL API Server] Load model error: {error_detail}")
        import sys
        sys.stderr.write(f"[Qwen3VL API Server] Load model error details:\n{error_detail}\n")
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail=error_detail)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnifiedReward-2.0-qwen3vl-8b Video Evaluation API Server')
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on (default: 34575 + RANK)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--base-port', type=int, default=34575, help='Base port number (default: 34575, actual port = base_port + RANK)')
    
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
    print(f"[RANK {rank}] API endpoint: http://{args.host}:{port}/evaluate_video")
    print(f"[RANK {rank}] API docs: http://{args.host}:{port}/docs")
    
    uvicorn.run(app, host=args.host, port=port)

