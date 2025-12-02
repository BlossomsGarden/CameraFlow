import requests
import re
import time

# face-1: 3.22
# face-2: 3.42
# face-3: 3.37
# face-4: 3.32

# 参数配置
video_path = "/home/ma-user/modelarts/user-job-dir/wlh/Code/UnifiedReward/face-4.mp4"
video_prompt = "A person stands near a train platform, positioned beside a large vertical sign. The individual is dressed in a patterned shirt and dark pants. The environment has industrial elements with concrete walls and visible train tracks. The lighting casts a warm glow, highlighting the textures and surfaces. The person remains mostly stationary, occasionally shifting posture."
fps = 30.0  # 视频的帧率
api_url = "http://localhost:34569/evaluate_video"

# 发送请求
payload = {
    "video_path": video_path,
    "prompt": video_prompt,
    "fps": fps
}

time_start = time.time()

while True:
    response = requests.post(api_url, json=payload, timeout=300)
    result = response.json()
    print(result)

    # 打印完整结果
    print("\nFull Evaluation Result:")
    print(result['output_text'])

    # 解析 <answer> 标签中的内容
    def extract_answer(text):
        """从文本中提取 <answer> 标签中的内容"""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # 提取并打印最终分数
    answer_content = extract_answer(result['output_text'])
    if answer_content:
        print("\n" + "="*50)
        print("Final Score (extracted from <answer> tag):")
        print(answer_content)
        print("="*50)
        break
    else:
        print("\nWarning: Could not find <answer> tag in the response.")

time_end = time.time()
print(f"Time cost: {time_end - time_start} seconds")

