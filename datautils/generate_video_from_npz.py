import os
import shutil
import subprocess
from npz2npy import extract_joints_from_npz_file

def batch_process_and_render(output_npy_folder, final_video_dir, render_path):

    # 第二步：调用 Blender 渲染命令
    print("\n🎬 正在调用 Blender 渲染视频...")
    blender_command = [
        "blender", "--background", "--python", render_path+"/render.py", "--",
        "--cfg="+render_path+"/configs/render_mld.yaml",
        f"--dir={output_npy_folder}",
        "--mode=video",
        "--joint_type=HumanML3D"
    ]
    subprocess.run(blender_command, cwd=render_path)

    # 第三步：移动渲染后的视频
    rendered_video_dir = output_npy_folder
    os.makedirs(final_video_dir, exist_ok=True)

    for file in os.listdir(rendered_video_dir):
        if file.endswith(".mp4"):
            src = os.path.join(rendered_video_dir, file)
            dst = os.path.join(final_video_dir, file)
            shutil.move(src, dst)
            print(f"✅ 已移动视频: {file} -> {final_video_dir}")

    print("🎉 全部完成！")

if __name__ == "__main__":
    output_npy_folder = "/root/autodl-tmp/CoT-data/motion_joint_data"  # 输出 .npy 目录
    final_video_dir = "/root/autodl-tmp/CoT-data/videos"
    render_path = "/root/autodl-fs/motion-latent-diffusion-main"

    batch_process_and_render(output_npy_folder, final_video_dir, render_path)
