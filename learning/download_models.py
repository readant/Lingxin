"""
MediaPipe模型下载脚本

下载新版Task API所需的模型文件到本地。
"""

import os
import urllib.request
import sys

# 模型下载URL
MODELS = {
    'hand_landmarker': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_lite': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}

def download_model(model_name, url, save_dir):
    """下载模型文件"""
    filename = f"{model_name}.task"
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print("[INFO] %s 已存在，跳过下载" % filename)
        return save_path

    print("[DOWNLOAD] 正在下载 %s..." % filename)
    try:
        urllib.request.urlretrieve(url, save_path)
        print("[SUCCESS] %s 下载完成: %s" % (filename, save_path))
        return save_path
    except Exception as e:
        print("[ERROR] 下载失败: %s" % str(e))
        return None

def main():
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')

    # 创建模型目录
    os.makedirs(models_dir, exist_ok=True)
    print("[INFO] 模型保存目录: %s\n" % models_dir)

    # 删除旧的模型文件（如果存在）
    old_pose_model = os.path.join(models_dir, 'pose_landmarker.task')
    if os.path.exists(old_pose_model):
        os.remove(old_pose_model)
        print("[INFO] 已删除旧模型文件: %s" % old_pose_model)

    # 下载所有模型
    for model_name, url in MODELS.items():
        download_model(model_name, url, models_dir)
        print()

    print("=" * 50)
    print("[SUCCESS] 模型下载完成！")
    print("模型保存在: %s" % models_dir)
    print("=" * 50)

if __name__ == '__main__':
    main()