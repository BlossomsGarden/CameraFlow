# conda create -n qwen3vl --clone PyTorch-2.1.0
# pip install torch==2.6.0 torch_npu==2.6.0 torchvision==0.21.0
# pip install av transformers==4.57.0

# ASCEND_RT_VISIBLE_DEVICES="0" python infer_unifiedreward_qwen3vl_8b.py

import torch
import torch_npu
import warnings
import time
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

# 加载 UnifiedReward-Think-qwen3vl-8b 模型
model_path = '/home/ma-user/modelarts/user-job-dir/wlh/Model/CodeGoat24/UnifiedReward-Think-qwen3vl-8b'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path)

# 视频和提示词配置
video_prompt = "A man and a woman are dancing together on a city street at dusk. The woman is wearing a yellow dress with a floral pattern and black shoes, while the man is dressed in a white shirt, dark tie, and dark pants with black shoes. They are both moving gracefully to the rhythm of the music, with the woman twirling and the man following her steps. The background features a cityscape with buildings and a distant mountain range, illuminated by the setting sun, creating a warm and romantic atmosphere."
video_path = '1-bad-10000.mp4'

# 设置视频的 fps 参数
fps = 30.0

# 评估模型的提示词
"""
model_prompt = f'''You are an objective and precise evaluator for video quality assessment. I will provide you with a text caption and a generated video based on that caption. The video is generated from a UE5 virtual scene (synthetic but realistic-looking), so you should NOT evaluate realism or authenticity - focus only on content quality and consistency. You must analyze this video carefully and provide a quality score.

        Instructions (MUST follow strictly):
        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.
        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
        3. After </think>, output the final score strictly inside <answer> and </answer> tags, containing only a single number in the format "X.X" where X.X is between 0.0 and 5.0.
        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.

        Evaluation procedure:

        1. The caption for the generated video is: 「{video_prompt}」. The provided video represents a candidate video that needs to be evaluated.

        2. You must evaluate the video across these four core dimensions:
        - Frame-to-Frame Consistency: Check if objects, characters, and scene elements maintain consistent shapes, colors, textures, and positions across frames. Assess whether objects remain stable when camera viewpoint changes. Look for flickering, jittering, or sudden appearance/disappearance of objects. Evaluate if character appearances (facial features, body proportions, clothing) remain consistent throughout the video.
        - Visual Quality and Artifacts: Check for black borders, edge artifacts, or incomplete frames. Identify if there are any visual artifacts or distortions. Assess if characters' body shapes are properly rendered without deformation or collapse. Check if objects maintain their proper shapes without distortion or morphing.
        - Prompt Alignment: Assess how well the video content matches the provided text prompt. Check if described objects, characters, actions, and scene elements are present and correctly depicted. Evaluate if the relationships between elements match the prompt description. Verify that attributes (colors, clothing, poses, etc.) align with the prompt.
        - Scene Coherence: Evaluate the smoothness and naturalness of character movements and animations. Check if the scene maintains logical spatial relationships throughout the video. Assess if lighting and shadows remain consistent across frames. Evaluate if background elements remain stable and coherent.

        3. For each evaluation dimension:
        - Provide a score between 0.0 and 5.0 (in increments of 0.1, e.g., 0.0, 0.1, 0.2, ..., 4.9, 5.0). NOT 0-1, NOT 0-10, NOT any other scale!
        - The score should reflect the overall quality considering all criteria, 
        - Lower scores (1.0-3.0) for videos with significant issues (artifacts, inconsistencies, poor alignment)
        - Higher scores (3.0-5.0) for high-quality, consistent, and well-aligned videos.
        - Provide a short rationale for the score (2–5 short sentences).
        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:
            N. Dimension name: 
                Score (x.x/5.0) - rationale

        4. After evaluating all dimensions, calculate the average score and show the calculation explicitly, following this exact format:
            Average score:
            (x.x + x.x + x.x + x.x) / 4 = final_score

        5. All reasoning, analysis, scoring, and calculations must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.

        6. The final score in <answer> must be the average of the four dimension scores, rounded to one decimal place (e.g., 3.5, 2.8, 4.1).

        Required output format (follow this exactly, including line breaks and indentation):

        <think>
        1. Frame-to-Frame Consistency: 
            Score (4.2/5.0) - ...
        2. Visual Quality and Artifacts: 
            Score (3.8/5.0) - ...
        3. Prompt Alignment: 
            Score (4.5/5.0) - ...
        4. Scene Coherence: 
            Score (4.0/5.0) - ...
        Average score:
        (4.2 + 3.8 + 4.5 + 4.0) / 4 = 4.1
        </think>
        <answer>4.1</answer>
    
        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly.'''
"""

model_prompt = f'''You are an objective and precise evaluator for video quality assessment. I will provide you with a text caption and a generated video based on that caption. The video is generated from a UE5 virtual scene (synthetic but realistic-looking), so you should NOT evaluate realism or authenticity - focus only on content quality and consistency. You must analyze this video carefully and provide a quality score. Do not provide any additional text.
        Evaluation procedure:

        1. The caption for the video is: 「{video_prompt}」. The provided video is the video that needs to be evaluated.

        2. You must evaluate the video across these four core dimensions:
        - Frame-to-Frame Consistency: Check if objects, characters, and scene elements maintain consistent shapes, colors, textures, and positions across frames. Assess whether objects remain stable when camera viewpoint changes. Look for flickering, jittering, or sudden appearance/disappearance of objects. Evaluate if character appearances (facial features, body proportions, clothing) remain consistent throughout the video.
        - Visual Quality and Artifacts: Check for black borders, edge artifacts, or incomplete frames. Identify if there are any visual artifacts or distortions. Assess if characters' body shapes are properly rendered without deformation or collapse. Check if objects maintain their proper shapes without distortion or morphing.
        - Prompt Alignment: Assess how well the video content matches the provided text prompt. Check if described objects, characters, actions, and scene elements are present and correctly depicted. Evaluate if the relationships between elements match the prompt description. Verify that attributes (colors, clothing, poses, etc.) align with the prompt.
        - Scene Coherence: Evaluate the smoothness and naturalness of character movements and animations. Check if the scene maintains logical spatial relationships throughout the video. Assess if lighting and shadows remain consistent across frames. Evaluate if background elements remain stable and coherent.

        3. For each evaluation dimension:
        - Provide a score between 1.0 and 5.0 (in increments of 0.1, e.g., 1.0, 1.1, 1.2, ..., 4.9, 5.0). NOT 0-1, NOT 0-10, NOT any other scale!
        - The score should reflect the overall quality considering all criteria, 
        - Lower scores (1.0-3.0) for videos with significant issues (artifacts, inconsistencies, poor alignment)
        - Higher scores (3.0-5.0) for high-quality, consistent, and well-aligned videos.
        - Do not provide any rationale.

        4. After evaluating all dimensions, calculate the average score of the four dimensions above as the final score.

        5. The final score must be the average of the four dimension scores, rounded to one decimal place (e.g., 3.5, 2.8, 4.1). Do not provide any additional text.

        Required output format, follow this exactly and do not provide any additional text or explanation:

            Final Score: X.X (where X.X is between 1.0 and 5.0)
    
        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly.'''


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

time_start = time.time()

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
    # max_new_tokens=256 意味着模型生成的最大新 token 数为 256
    generated_ids = model.generate(**inputs, max_new_tokens=128)
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

print(output_text)

time_end = time.time()
print(f"Time cost: {time_end - time_start} seconds")

