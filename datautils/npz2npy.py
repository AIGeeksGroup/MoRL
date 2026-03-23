import numpy as np
import os


def extract_joints_from_npz(in_dir, out_dir):
    """
    从指定文件夹中的 .npz 文件中提取 'joints' 字段并保存为 .npy 文件。
    :param in_dir: 输入目录（包含 .npz 文件）
    :param out_dir: 输出目录（保存 .npy 文件）
    """
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if file.endswith(".npz"):
            path = os.path.join(in_dir, file)
            npz = np.load(path, allow_pickle=True)
            if "joints" in npz:
                joints = npz["joints"]
                out_path = os.path.join(out_dir, file.replace(".npz", ".npy"))
                np.save(out_path, joints)
                print(f"Saved joints to: {out_path}")
            else:
                print(f"[WARN] 'joints' key not found in {file}")


def extract_joints_from_npz_file(npz_path, output_dir):
    """
    从单个 .npz 文件中提取 'joints' 字段并保存为同名的 .npy 文件到指定目录。

    :param npz_path: 输入的 .npz 文件路径
    :param output_dir: 输出目录（保存 .npy 文件）
    """
    os.makedirs(output_dir, exist_ok=True)

    if not npz_path.endswith(".npz"):
        print(f"❌ 错误: 输入文件必须是 .npz 文件: {npz_path}")
        return

    try:
        data = np.load(npz_path, allow_pickle=True)
        if "joints" not in data:
            print(f"⚠️ 警告: 文件中未找到 'joints' 字段: {npz_path}")
            return

        joints = data["joints"]
        filename = os.path.splitext(os.path.basename(npz_path))[0] + ".npy"
        output_path = os.path.join(output_dir, filename)

        np.save(output_path, joints)
        print(f"✅ 已保存 joints 到: {output_path}")
    except Exception as e:
        print(f"❌ 处理失败: {npz_path}")
        print(e)


if __name__ == "__main__":
    # 示例路径（你可以修改）
    input_npz = "/root/autodl-tmp/motionx/motion_data/standard_smplx/aist/subset_0000/Dance_Break_3_Step_clip_1.npz"
    output_dir = "/root/autodl-tmp/CoT-data/motion_data"

    extract_joints_from_npz_file(input_npz, output_dir)

