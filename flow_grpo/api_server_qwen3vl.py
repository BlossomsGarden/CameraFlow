"""
API server for UnifiedReward-Think-qwen3vl-8b video evaluation.
Run with: python api_server_qwen3vl.py --port 34569

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

app = FastAPI(title="UnifiedReward-Think-qwen3vl-8b Video Evaluation API")

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

# Default model path
default_model_path = '/home/ma-user/modelarts/user-job-dir/wlh/Model/CodeGoat24/UnifiedReward-Think-qwen3vl-8b'

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
    model_prompt = f'''You are an objective and precise evaluator for video quality assessment. I will provide you with a text caption and a generated video based on that caption. The video is generated from a UE5 virtual scene (synthetic but realistic-looking), so you should NOT evaluate realism or authenticity - focus only on content quality and consistency. You must analyze this video carefully and provide a quality score.

        CRITICAL SCORING RULES (MUST FOLLOW STRICTLY):
        1. ALL scores MUST be between 1.0 and 5.0 ONLY. NOT 0.0, NOT 0-1, NOT 0-10, NOT any other scale!
        2. ALL scores MUST be in increments of 0.1 (e.g., 1.0, 1.1, 1.2, ..., 4.9, 5.0)
        3. When writing scores in <think>, use format "Score: X.X" where X.X is between 1.0 and 5.0. DO NOT use "/5", "/10", "/0", or any other denominator.
        4. The <answer> tag is MANDATORY and MUST ALWAYS be present in your output. You CANNOT skip or omit the <answer> tag.
        5. The <answer> tag MUST contain ONLY a single number in format "X.X" (ONE decimal place) where X.X is between 1.0 and 5.0. Examples: 3.5, 4.2, 2.8, 3.1. NOT 3.45, NOT 3.07, NOT multiple decimal places - EXACTLY ONE decimal place!
        6. NO text, NO calculations, NO "/5", NO "/10", NO equations, NO "average:", NO other words in <answer> - ONLY the number with ONE decimal place.

        Instructions (MUST follow strictly):
        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.
        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
        3. After </think>, you MUST ALWAYS output the final score inside <answer> and </answer> tags. The <answer> tag is MANDATORY - you CANNOT skip it or omit it.
        4. The <answer> tag MUST contain ONLY a number between 1.0 and 5.0 in format "X.X" with EXACTLY ONE decimal place (e.g., 3.5, 4.2, 2.8, 3.1). NOT 3.45, NOT 3.07, NOT 3.125 - EXACTLY ONE decimal place!
        5. NO other text, NO calculations, NO "/5", NO equations, NO "average:" in <answer> - ONLY the number with ONE decimal place.
        6. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.

        Evaluation procedure:

        1. The caption for the generated video is: 「{video_prompt}」. The provided video represents a candidate video that needs to be evaluated.

        2. You must evaluate the video across these four core dimensions:
        - Frame-to-Frame Consistency: Check if objects, characters, and scene elements maintain consistent shapes, colors, textures, and positions across frames. Assess whether objects remain stable when camera viewpoint changes. Look for flickering, jittering, or sudden appearance/disappearance of objects. Evaluate if character appearances (facial features, body proportions, clothing) remain consistent throughout the video.
        - Visual Quality and Artifacts: Check for black borders, edge artifacts, or incomplete frames. Identify if there are any visual artifacts or distortions. Assess if characters' body shapes are properly rendered without deformation or collapse. Check if objects maintain their proper shapes without distortion or morphing.
        - Prompt Alignment: Assess how well the video content matches the provided text prompt. Check if described objects, characters, actions, and scene elements are present and correctly depicted. Evaluate if the relationships between elements match the prompt description. Verify that attributes (colors, clothing, poses, etc.) align with the prompt.
        - Scene Coherence: Evaluate the smoothness and naturalness of character movements and animations. Check if the scene maintains logical spatial relationships throughout the video. Assess if lighting and shadows remain consistent across frames. Evaluate if background elements remain stable and coherent.

        3. For each evaluation dimension (1, 2, 3, 4):
        - Provide a score between 1.0 and 5.0. NOT 0.0, NOT 0-1, NOT 0-10, NOT any other scale!
        - The score should reflect the overall quality considering all criteria
        - Lower scores (1.0-3.0) for videos with significant issues (artifacts, inconsistencies, poor alignment)
        - Higher scores (3.0-5.0) for high-quality, consistent, and well-aligned videos
        - Provide a BRIEF rationale for the score: ONLY 2-3 SHORT sentences. Do NOT use numbered lists (#1, #2, #3). Do NOT write long paragraphs. Keep it concise.
        - Each dimension must follow exactly this format with numbering, line breaks, and indentation:
            N. Dimension name: 
                Score: X.X - A Brief rationale (2-3 short sentences only, one line only, no numbered lists, no long paragraphs)

        4. After evaluating all FOUR dimensions (1, 2, 3, 4), calculate the AVERAGE of these FOUR dimension scores. This is the average of the four main dimensions, NOT an average within each dimension.
        - Take the four scores from dimensions 1, 2, 3, and 4
        - Calculate: (Score1 + Score2 + Score3 + Score4) / 4
        - Show the calculation explicitly, following this EXACT format (write it ONCE, do NOT repeat):
            Average score:
            (X.X + X.X + X.X + X.X) / 4 = X.XX
        Note: 
        - The average is calculated from the FOUR main dimension scores (1, 2, 3, 4), NOT from sub-items within each dimension.
        - All individual dimension scores (X.X) must be between 1.0 and 5.0 with ONE decimal place.
        - The calculated average (X.XX) must be shown with EXACTLY TWO decimal places (e.g., 3.45, 4.20, 2.75, 3.48). NOT a long decimal like 1.3303339920043945 - round to TWO decimal places!
        - Write the calculation ONCE only. Do NOT repeat numbers or create loops.
        - Example: If scores are 3.3, 3.3, 4.0, 2.3, then average = (3.3 + 3.3 + 4.0 + 2.3) / 4 = 3.23

        5. All reasoning, analysis, scoring, and calculations must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.

        6. The Average score in <think> must show TWO decimal places (e.g., 3.45, 4.20, 2.75). Write the calculation ONCE as a single line: (X.X + X.X + X.X + X.X) / 4 = X.XX. Do NOT repeat numbers or create loops.
        7. The final score in <answer> must be the average rounded to EXACTLY ONE decimal place (e.g., 3.5, 4.2, 2.8, 3.1). If average is 3.45, round to 3.5. If average is 3.43, round to 3.4. NOT 3.45, NOT 3.07 - EXACTLY ONE decimal place!
        8. The <answer> tag is MANDATORY - you MUST ALWAYS include it. You CANNOT skip or omit the <answer> tag.
        9. The <answer> tag MUST contain ONLY the number with ONE decimal place. NO "/5", NO "/10", NO "average:", NO calculations, NO other text.

        Required output format (follow this exactly, including line breaks and indentation):

        <think>
        1. Frame-to-Frame Consistency: 
            Score: 4.2 - The video demonstrates excellent consistency across frames. Objects remain stable throughout.
        2. Visual Quality and Artifacts: 
            Score: 3.8 - The visual quality is high with minimal artifacts. Some minor distortions are present.
        3. Prompt Alignment: 
            Score: 4.5 - The video aligns very well with the prompt. Most elements are correctly depicted.
        4. Scene Coherence: 
            Score: 4.0 - The scene maintains logical relationships. Lighting and shadows are consistent.
        Average score:
        (4.2 + 3.8 + 4.5 + 4.0) / 4 = 4.13
        </think>
        <answer>4.1</answer>
        
        Note: 
        - Each dimension (1, 2, 3, 4) has ONE score with ONE decimal place (4.2, 3.8, 4.5, 4.0)
        - Each dimension rationale is BRIEF: only 2-3 short sentences. No numbered lists (#1, #2), no long paragraphs.
        - The average is calculated from the FOUR dimension scores: (4.2 + 3.8 + 4.5 + 4.0) / 4 = 4.13
        - The average calculation shows TWO decimal places (4.13), NOT a long decimal number
        - The <answer> tag contains the final score rounded to ONE decimal place (4.1)
        - Write the calculation ONCE only. Do NOT repeat or loop.

        CRITICAL REMINDERS (READ CAREFULLY):
        - ALL dimension scores (for dimensions 1, 2, 3, 4) must be between 1.0 and 5.0 with ONE decimal place (e.g., 3.3, 4.1, 2.8). NO scores outside this range.
        - Use "Score: X.X" format for each dimension in <think>. NO "/5", NO "/10", NO "/0".
        - Each dimension rationale must be BRIEF: only 2-3 short sentences. Do NOT use numbered lists (#1, #2, #3). Do NOT write long paragraphs or detailed analysis.
        - The Average score is the average of the FOUR main dimension scores (1, 2, 3, 4), NOT an average within each dimension.
        - The Average score calculation must show TWO decimal places (e.g., 3.45, 4.20, 2.75). NOT a long decimal like 1.3303339920043945 - round to TWO decimal places! Write the calculation ONCE only - do NOT repeat numbers or create loops.
        - The <answer> tag is MANDATORY - you MUST ALWAYS include it in your output. You CANNOT skip, omit, or forget the <answer> tag. Every response MUST end with <answer>X.X</answer>.
        - The <answer> tag MUST contain ONLY a number with EXACTLY ONE decimal place (e.g., 4.1, 3.5, 2.8, 3.1). This is the final score rounded from the two-decimal average. NOT 3.45, NOT 3.07 - EXACTLY ONE decimal place!
        - NO "/5", NO "/10", NO "average:", NO calculations, NO other text in <answer> - ONLY the number with ONE decimal place.
        - If you see any score outside 1.0-5.0 range, you MUST convert it to 1.0-5.0 range before using it.
        - Write the Average score calculation ONCE. Do NOT repeat numbers like "1.0, 2.0, 3.0, 4.0" multiple times. Write it as a single calculation line only: (X.X + X.X + X.X + X.X) / 4 = X.XX
    
        Note: The example above is only to illustrate the exact format. Your actual evaluation must follow this format exactly.'''
    
    return model_prompt

def load_model(model_dir: str = None):
    """
    Load model to GPU memory (thread-safe, only loads once).
    
    Args:
        model_dir: Model directory path (if None, uses default)
    
    Returns:
        tuple: (model, processor, device)
    """
    global _model, _processor, _model_path, _device
    
    if model_dir is None:
        model_dir = default_model_path
    
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
    Evaluate video quality and alignment with text caption using UnifiedReward-Think-qwen3vl-8b model.
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
        # 清理显存：安全删除所有中间变量（避免变量未定义错误）
        def safe_delete(var):
            try:
                del var
            except (NameError, UnboundLocalError):
                pass
            
        safe_delete(inputs)
        safe_delete(generated_ids)
        safe_delete(generated_ids_trimmed)
        # 清理NPU缓存
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    
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
        if model_dir is None:
            model_dir = default_model_path
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
    parser = argparse.ArgumentParser(description='UnifiedReward-Think-qwen3vl-8b Video Evaluation API Server')
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on (default: 34569 + RANK)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--base-port', type=int, default=34569, help='Base port number (default: 34569, actual port = base_port + RANK)')
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Path to the UnifiedReward-Think-qwen3vl-8b model')
    
    args = parser.parse_args()
    default_model_path = args.model_path
    
    # Get RANK from environment variable (for multi-GPU setup)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    
    # Calculate port: use specified port, or base_port + rank
    if args.port is not None:
        port = args.port
    else:
        port = args.base_port + rank
    
    print(f"[RANK {rank}] Starting API server on {args.host}:{port}")
    print(f"[RANK {rank}] Model path: {default_model_path}")
    print(f"[RANK {rank}] Health check: http://{args.host}:{port}/health")
    print(f"[RANK {rank}] API endpoint: http://{args.host}:{port}/evaluate_video")
    print(f"[RANK {rank}] API docs: http://{args.host}:{port}/docs")
    
    uvicorn.run(app, host=args.host, port=port)

