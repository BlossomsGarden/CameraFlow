# conda create -n qwen3vl --clone PyTorch-2.1.0
# pip install torch==2.6.0 torch_npu==2.6.0 torchvision==0.21.0
# pip install av transformers==4.57.0

# ASCEND_RT_VISIBLE_DEVICES="0" python qwen3-vl-8b-infer.py

import torch
import torch_npu
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/ma-user/modelarts/user-job-dir/wlh/Model/Qwen/Qwen3-VL-8B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("/home/ma-user/modelarts/user-job-dir/wlh/Model/Qwen/Qwen3-VL-8B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "0.mp4",
                "fps": 30.0
            },
            {"type": "text", "text": "Describe this video."},
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

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)