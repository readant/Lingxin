"""
MediaPipe模型下载脚本

下载新版Task API所需的模型文件到本地。
"""

import os
import socket
import urllib.request
import sys

# 模型下载 URL
MODELS = {
    'hand_landmarker': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose_landmarker_lite': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}

# 国内网络访问 Google 存储易超时，设置全局超时并做失败重试
socket.setdefaulttimeout(30)
MAX_RETRIES = 3
# 最小有效文件大小（字节），防止下载到错误页或残缺文件
MIN_FILE_SIZE = 1024 * 1024  # 1MB


def download_model(model_name, url, save_dir):
    """下载模型文件，失败自动重试"""
    filename = f"{model_name}.task"
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print("[INFO] %s 已存在，跳过下载" % filename)
        return save_path

    for attempt in range(1, MAX_RETRIES + 1):
        print("[DOWNLOAD] 正在下载 %s... (第 %d/%d 次尝试)" % (filename, attempt, MAX_RETRIES))
        try:
            urllib.request.urlretrieve(url, save_path)
            # 校验文件大小，防止下载到错误页或残缺文件
            if os.path.getsize(save_path) < MIN_FILE_SIZE:
                os.remove(save_path)
                print("[ERROR] %s 文件过小，可能下载失败" % filename)
            else:
                print("[SUCCESS] %s 下载完成: %s" % (filename, save_path))
                return save_path
        except Exception as e:
            # 清理残缺文件后重试
            if os.path.exists(save_path):
                os.remove(save_path)
            print("[ERROR] 下载失败: %s" % str(e))
    print("[ERROR] %s 多次尝试后仍失败，请检查网络后重新运行" % filename)
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
    print("\n下一步：运行 00_env_setup.py 检查环境")

if __name__ == '__main__':
    main()
