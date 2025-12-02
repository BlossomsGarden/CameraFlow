# conda create -name unifiedrewardqwen3 --clone PyTorch-2.1.0
# pip install transformers==4.52.4 qwen-vl-utils==0.0.14 
# pip install torch==2.6.0 torch_npu==2.6.0 torchvision==0.21.0
#
# ASCEND_RT_VISIBLE_DEVICES="0" python infer_qwen2.5vl_7b_demo.py

# face-1: 3.15
# face-2: 3.30
# face-3: 3.15
# face-4: 

import torch
import torch_npu
import os
import json
import random
from PIL import Image
from tqdm import trange
import warnings

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = '/home/ma-user/modelarts/user-job-dir/wlh/Model/CodeGoat24/UnifiedReward-2.0-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

video_path = "/home/ma-user/modelarts/user-job-dir/wlh/Code/UnifiedReward/face-4.mp4"
video_prompt = "A person stands near a train platform, positioned beside a large vertical sign. The individual is dressed in a patterned shirt and dark pants. The environment has industrial elements with concrete walls and visible train tracks. The lighting casts a warm glow, highlighting the textures and surfaces. The person remains mostly stationary, occasionally shifting posture."


# 设置视频的 fps 参数
fps = 30.0

# # 评估模型的提示词
# model_prompt = (
#     "\nYou are given a text prompt and a generated video based on that prompt. "
#     "Your task is to evaluate this video generation quality for reinforcement learning reward. "
#     "The video is generated from a UE5 virtual scene (synthetic but realistic-looking), "
#     "so you should NOT evaluate realism or authenticity - focus only on content quality and consistency.\n\n"
    
#     "Evaluate the video based on the following criteria:\n\n"
    
#     "1. Frame-to-Frame Consistency:\n"
#     "   - Check if objects, characters, and scene elements maintain consistent shapes, colors, textures, and positions across frames\n"
#     "   - Assess whether objects remain stable when camera viewpoint changes\n"
#     "   - Look for flickering, jittering, or sudden appearance/disappearance of objects\n"
#     "   - Evaluate if character appearances (facial features, body proportions, clothing) remain consistent throughout the video\n\n"
    
#     "2. Visual Quality and Artifacts:\n"
#     "   - Check for black borders, edge artifacts, or incomplete frames\n"
#     "   - Identify if there are any visual artifacts or distortions\n"
#     "   - Assess if characters' body shapes is properly rendered without deformation or collapse\n"
#     "   - Check if objects maintain their proper shapes without distortion or morphing\n\n"

#     "3. Prompt Alignment:\n"
#     "   - Assess how well the video content matches the provided text prompt\n"
#     "   - Check if described objects, characters, actions, and scene elements are present and correctly depicted\n"
#     "   - Evaluate if the relationships between elements match the prompt description\n"
#     "   - Verify that attributes (colors, clothing, poses, etc.) align with the prompt\n\n"
    
#     "4. Scene Coherence:\n"
#     "   - Evaluate the smoothness and naturalness of character movements and animations\n"
#     "   - Check if the scene maintains logical spatial relationships throughout the video\n"
#     "   - Assess if lighting and shadows remain consistent across frames\n"
#     "   - Evaluate if background elements remain stable and coherent\n\n"
    
#     "IMPORTANT SCORING REQUIREMENTS:\n"
#     "- The score MUST be between 1.0 and 5.0 (NOT 0-5, NOT 0-10, NOT any other scale)\n"
#     "- Valid scores are: 1.0, 1.1, ..., 2.1, 2.2, ..., 5.0 (in increments of 0.1)\n"
#     "- The score should reflect the overall quality considering all criteria, "
#     "- Lower scores (1.0-3.0) for videos with significant issues (artifacts, inconsistencies, poor alignment) "
#     "- Higher scores (3.0-5.0) for high-quality, consistent, and well-aligned videos. Get the average score of the four dimensions as the final score.\n\n"
    
#     "OUTPUT FORMAT REQUIREMENTS:\n"
#     "- You MUST only provide a single Final Score like 1.8 or 3.5. Do NOT use any other format or any additional text.\n"
#     "- The output must be only 'X.X' where X.X is in the range [1.0, 5.0]\n\n"
    
#     f"Text Prompt: [{video_prompt}]\n\n"
    
#     "Please provide your evaluation and then output the Final Score in the required format."
# )

# model_prompt = (
#     "You are presented with a generated video and its associated text caption. "
#     "Your task is to analyze the video across multiple dimensions in relation to the caption. Specifically:\n"
#     "Provide overall assessments for the video along the following axes (each rated from 1 to 5):\n"
#     "- Alignment Score: How well the video matches the caption in terms of content.\n"
#     "- Physics Score: How well the gravity, movements, collisions, and interactions make physical sense.\n"
#     "- Style Score: How visually appealing the video looks, regardless of caption accuracy.\n\n"
#     "Output your evaluation using the format below:\n\n"
#     "Alignment Score (1-5): X\n"
#     "Physics Score (1-5): Y\n"
#     "Style Score (1-5): Z\n\n"
#     "Your task is provided as follows:\n"
#     f"Text Caption: [{video_prompt}]"
# )


model_prompt = f'''You are an objective and precise evaluator for video quality assessment. I will provide you with a text caption and a generated video based on that caption. Evaluate the video based on the caption: 「{video_prompt}」of four dimensions I provid and then give me the average score. The video is from a UE5 virtual scene (synthetic but realistic-looking), so focus on content quality and consistency, not realism or authenticity.

    TOTAL FOUR EVALUATION DIMENSIONS (You MUST evaluate the video across these four core dimensions STRICTLY):
    You must evaluate the video across these four core dimensions:
    Dimension1: Frame-to-Frame Consistency: Check if objects, characters, and scene elements maintain consistent shapes, colors, textures, and positions across frames. Look for flickering, jittering, or sudden appearance/disappearance.
    Dimension2: Visual Quality and Artifacts: Check for black borders, edge artifacts, incomplete frames, distortions, or deformation of objects/characters.
    Dimension3: Prompt Alignment: Assess how well the video matches the text prompt. Check if described objects, characters, actions, and scene elements are present and correctly depicted.
    Dimension4: Scene Coherence: Evaluate smoothness of movements, logical spatial relationships, consistent lighting/shadows, and stable background elements.


    For each of the four evaluation dimensions mentioned above, get Score1, Score2, Score3, Score4 respectively:
    - Score1, Score2, Score3, Score4 (in a format "X.X") must be between 1.0 and 5.0 with one decimal place (e.g., 3.5, 4.2, 2.8). NOT 0-1, NOT 0-10, NOT any other scale!
    - Lower scores (1.0-3.0) for videos with significant issues (artifacts, inconsistencies, poor alignment). Higher scores (3.0-5.0) for high-quality, consistent, and well-aligned videos
    - Provide a BRIEF rationale for the score: ONLY 2-3 SHORT sentences. Only one line. Do NOT use numbered lists (#1, #2, #3). Do NOT write long paragraphs. Keep it concise!
    - Each of the four dimension must follow exactly this format with numbering, line breaks, and indentation:
        1. name: 
            Score1: X.X - A Brief rationale...

    
    After evaluating all FOUR dimensions mentioned above, then calculate the Average score as the final score:
    - Take the four scores Score1, Score2, Score3, Score4 from Dimensions 1, 2, 3, and 4, then calculate: (Score1 + Score2 + Score3 + Score4) / 4
    - The average is calculated from the FOUR main dimension scores Score1, Score2, Score3, Score4, NOT from sub-items within each dimension or other scores.
    - The calculated average (X.XX) must be shown with EXACTLY TWO decimal places (e.g., 3.45, 4.20, 2.75, 3.48). NOT a long decimal like 1.3303339920043945 - round to TWO decimal places!
    - Write the calculation ONCE only. Do NOT repeat numbers or create loops.
    - **CRITICAL: You MUST show the calculation explicitly INSIDE <think> BEFORE closing the tag, following this EXACT format (write it ONCE, do NOT repeat):**
        Average score:
        (X.X + X.X + X.X + X.X) / 4 = X.XX
    - **MANDATORY: The calculation formula MUST appear inside <think> tags. You CANNOT close </think> without showing the calculation process.**


    Instructions (MUST FOLLOW STRICTLY):
    1.  All reasoning, analysis, scoring, and calculations must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.
    2. The <think> block must start immediately with the first evaluation dimension mentioned above. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
    3. **MANDATORY STEP: Before closing </think>, you MUST write the Average score calculation formula inside <think> following the exact format:**
        Average score:
        (Score1 + Score2 + Score3 + Score4) / 4 = X.XX
        **You CANNOT close </think> without showing this calculation process.**
    4. After </think>, you MUST ALWAYS output the final average score inside <answer> and </answer> tags. The <answer></answer> tag is MANDATORY - you CANNOT skip it or omit it.
    5. **CRITICAL: The number inside <answer></answer> MUST be EXACTLY the average of Score1, Score2, Score3, Score4 that you calculated before. It MUST be (Score1 + Score2 + Score3 + Score4) / 4. You CANNOT use any other number or arbitrary value. The <answer> value MUST match the calculated average from the formula shown in <think>.**
    6. The <answer> tag MUST contain ONLY the number with TWO decimal place like "X.XX"（e.g. 3.45, 4.20). NO other text, NO calculations, NO equations, NO "average:", NO other words in <answer>. ONLY the number with TWO decimal place.
    7. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.
    

    Required OUTPUT FORMAT Example (MUST FOLLOW STRICTLY, including line breaks and indentation):

    <think>
    Dimension 1: Frame-to-Frame Consistency: 
        Score1: 3.3 - Brief rationale... (2-3 sentences)
    Dimension 2: Visual Quality and Artifacts: 
        Score2: 3.3 - Brief rationale... (2-3 sentences)
    Dimension 3: Prompt Alignment: 
        Score3: 4.0 - Brief rationale... (2-3 sentences)
    Dimension 4: Scene Coherence: 
        Score4: 2.6 - Brief rationale... (2-3 sentences)
    Average score:
    (3.3 + 3.3 + 4.0 + 2.6) / 4 = 3.30
    </think>
    <answer>3.30</answer>

    Note: The example numbers above are only to illustrate the exact format. Your actual evaluation must follow this format exactly STRICTLY.

    CRITICAL REMINDERS (READ CAREFULLY):
    - ALL dimension scores Score1, Score2, Score3, Score4 must be between 1.0 and 5.0 with ONE decimal place (e.g., 3.3, 4.1, 2.8). NO scores outside this range.
    - Each dimension rationale must be BRIEF: only 2-3 short sentences. Do NOT use numbered lists (#1, #2, #3). Do NOT write long paragraphs or detailed analysis.
    - **MANDATORY: You MUST write the Average score calculation formula INSIDE <think> BEFORE closing the tag. The format MUST be:**
        Average score:
        (X.X + X.X + X.X + X.X) / 4 = X.XX
        **You CANNOT skip this step or close </think> without showing the calculation.**
    - Write the Average score calculation ONCE. Do NOT repeat numbers like "1.0, 2.0, 3.0, 4.0" multiple times when you give Average score calculation. Write it as a single calculation line only like: "(X.X + X.X + X.X + X.X) / 4 = X.XX"
    - The Average score is the average of the FOUR main dimension scores Score1, Score2, Score3, Score4, NOT an average within each dimension or other scores.
    - The Average score calculation result must show TWO decimal places (e.g., 3.45, 4.20, 2.75). NOT a long decimal like 1.3303339920043945 - round to TWO decimal places! Write the calculation ONCE only - do NOT repeat numbers or create loops.
    - **CRITICAL: The number inside <answer></answer> MUST be EXACTLY equal to (Score1 + Score2 + Score3 + Score4) / 4. It MUST be the calculated average from the four dimension scores you evaluated. You CANNOT use any arbitrary number, random value, or different calculation. The <answer> value MUST match the average you calculated and showed in the formula inside <think>.**
    - The <answer> </answer> tag MUST contain ONLY the number with TWO decimal place like "X.XX"（e.g. 3.45, 4.20). NO other text, NO calculations, NO equations, NO "average:", NO other words in <answer>. ONLY the number with TWO decimal place.
    '''

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": fps},
            {"type": "text", "text": model_prompt}
        ]
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
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
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

print(output_text)