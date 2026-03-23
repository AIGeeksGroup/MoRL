import os
import base64
import json
import requests


def generate_cot_from_video_and_caption(video_path, caption_path, output_dir, api_key,
                                        endpoint="https://yansd666.top/v1beta/models/gemini-2.5-pro:generateContent"):
    """
    给定视频路径和对应的 caption 路径，生成 CoT 推理并保存为 output_dir 中与视频同名的 .txt 文件。
    """

    # === Step 1: 检查文件是否存在 ===
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    if not os.path.exists(caption_path):
        print(f"❌ Caption 文件不存在: {caption_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # === Step 2: 读取 caption 内容 ===
    with open(caption_path, "r", encoding="utf-8") as f:
        caption_text = f.read().strip()

    # === Step 3: 读取视频并转为 base64 ===
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_base64 = base64.b64encode(video_data).decode("utf-8")

    # === Step 4: 构造 prompt ===
    prompt = f"""
You are a visual assistant specialized in understanding 3D human motion sequences. For each video, you will be given the following caption: {caption_text}

Then, please carefully observe and analyze the motion in this video. Think step by step like a human watching and reasoning about what the person is doing. Break down the complex action into smaller steps, describe them in order, and try to understand the purpose or pattern. 

Use internal thoughts like "first", "then", "next", "maybe", "it seems", etc. 

You are encouraged to reflect and verify your guess during the reasoning.

Please just provide your detailed reasoning between the <think> and </think> tags, and your final answer between <answer> and </answer> tags.

Question: What's the human doing in the video?

Output format:
<think>...reasoning ...</think>
<answer>...final answer...</answer>
"""

    # === Step 5: 构造请求体 ===
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "video/mp4",
                        "data": video_base64
                    }
                }
            ]
        }]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # === Step 6: 发送请求 ===
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"🚀 正在处理视频：{video_name}")
    response = requests.post(endpoint, headers=headers, data=json.dumps(payload))

    # === Step 7: 处理响应 ===
    if response.status_code == 200:
        try:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            output_path = os.path.join(output_dir, f"{video_name}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ CoT 已保存到：{output_path}")
        except Exception as e:
            print("❌ 无法解析响应内容：")
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"❌ 请求失败：状态码 {response.status_code}")
        print(response.text)


def batch_generate_cot(video_dir, caption_dir, output_dir, api_key):
    """
    批量对视频生成 CoT 数据
    :param video_dir: 视频文件夹 A
    :param caption_dir: Caption 文件夹 B
    :param output_dir: 输出 CoT 文件夹 C
    :param api_key: Gemini API Key
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in sorted(os.listdir(video_dir)):
        if file.endswith(".mp4"):
            video_path = os.path.join(video_dir, file)
            filename_wo_ext = os.path.splitext(file)[0]
            caption_path = os.path.join(caption_dir, f"{filename_wo_ext}.txt")

            if os.path.exists(caption_path):
                try:
                    generate_cot_from_video_and_caption(
                        video_path=video_path,
                        caption_path=caption_path,
                        output_dir=output_dir,
                        api_key=api_key
                    )
                except Exception as e:
                    print(f"❌ 处理 {file} 时出错: {e}")
            else:
                print(f"⚠️ 找不到对应的 caption：{caption_path}")


if __name__ == "__main__":
    # ✅ 配置路径
    video_dir = "/root/autodl-tmp/CoT-data/videos"  # 文件夹 A：渲染后的视频
    caption_dir = "/root/autodl-tmp/motionx/new_caption/aist/subset_0000"  # 文件夹 B：包含 Caption 的 txt 文件
    output_dir = "/root/autodl-tmp/CoT-data/CoT"  # 文件夹 C：CoT 输出目录
    api_key = "your_api_key"  # ✅ 替换为你的 API Key

    batch_generate_cot(video_dir, caption_dir, output_dir, api_key)

# # === 示例使用 ===
# if __name__ == "__main__":
#     video_path = "/root/autodl-tmp/motionx/motion_data/standard_smplx/aist/test/1.mp4"
#     caption_path = "/root/autodl-tmp/motionx/new_caption/aist/subset_0000/Dance_Break_3_Step_clip_1.txt"
#     output_dir = "/root/autodl-tmp/CoT-data/CoT"  # ✅ 输出文件夹
#     api_key = "key"

#     generate_cot_from_video_and_caption(video_path, caption_path, output_dir, api_key)

