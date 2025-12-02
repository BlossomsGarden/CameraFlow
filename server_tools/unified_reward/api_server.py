"""
API server for UnifiedReward-Think-qwen3vl-8b video evaluation.
Run with: python api_server_qwen3vl.py --port 34569

ASCEND_RT_VISIBLE_DEVICES="0" python api_server_qwen3vl.py
"""
import argparse
import warnings
import torch
import torch_npu
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import uvicorn

warnings.filterwarnings("ignore")

app = FastAPI(title="UnifiedReward-Think-qwen3vl-8b Video Evaluation API")

# Global model and processor (loaded once at startup)
model = None
processor = None
model_path = '/home/ma-user/modelarts/user-job-dir/wlh/Model/CodeGoat24/UnifiedReward-2.0-qwen3vl-8b'

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
            - Each of the four dimension must follow exactly this format with numbering, line breaks, and indentation:
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

def evaluate_video_quality(video_path: str, prompt: str, fps: float = 30.0) -> str:
    """
    Evaluate video quality and alignment with text caption using UnifiedReward-Think-qwen3vl-8b model.
    
    Args:
        video_path: Path to the video file
        prompt: Text caption describing the video content
        fps: Frame rate for video processing (default: 30.0)
    
    Returns:
        Evaluation output text containing the score
    """
    global model, processor
    
    if model is None or processor is None:
        raise RuntimeError("Model not loaded. Please ensure model_path is correct.")
    
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

@app.post('/evaluate_video', response_model=EvaluationResponse)
async def evaluate_video(request: EvaluationRequest):
    """
    Evaluate video quality and alignment with text caption.
    
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
        # Call evaluation function
        output_text = evaluate_video_quality(
            video_path=request.video_path,
            prompt=request.prompt,
            fps=request.fps
        )
        
        # Return results
        return EvaluationResponse(
            output_text=output_text
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UnifiedReward-Think-qwen3vl-8b Video Evaluation API Server')
    parser.add_argument('--port', type=int, default=34569, help='Port to run the server on (default: 34569)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--model_path', type=str, default=model_path, help='Path to the UnifiedReward-Think-qwen3vl-8b model')
    
    args = parser.parse_args()
    model_path = args.model_path
    
    # 启动时直接加载模型
    print(f"Loading model from {model_path}...")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Model path: {model_path}")
    print(f"API endpoint: http://{args.host}:{args.port}/evaluate_video")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port)

