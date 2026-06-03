"""
第0阶段：环境准备 - 环境检查脚本

本脚本用于检查Python环境和项目依赖是否正确安装。
"""

import sys
import subprocess

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"[检查] Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and 9 <= version.minor <= 11:
        print("[OK] Python版本符合要求 (3.9-3.11)")
        return True
    else:
        print("[ERROR] Python版本不符合要求，请安装Python 3.9-3.11")
        return False

def check_package_installed(package_name, min_version=None):
    """检查包是否安装"""
    try:
        result = subprocess.run(
            [sys.executable, '-c', f"import {package_name}; print({package_name}.__version__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[OK] {package_name} 已安装，版本: {version}")
            return True
        else:
            print(f"[ERROR] {package_name} 未安装")
            return False
    except Exception as e:
        print(f"[ERROR] 检查 {package_name} 时出错: {e}")
        return False

def main():
    print("=" * 60)
    print("聆心项目 - 环境检查")
    print("=" * 60)

    # 检查Python版本
    if not check_python_version():
        return

    print("\n[开始检查项目依赖...]")

    # 检查核心依赖
    packages = [
        'mediapipe',
        'cv2',
        'numpy',
        'pandas',
        'sklearn',
        'torch',
        'matplotlib',
        'tqdm'
    ]

    all_passed = True
    for package in packages:
        if not check_package_installed(package):
            all_passed = False

    # 检查模型文件
    print("\n[检查模型文件...]")
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, 'models')

    required_models = ['hand_landmarker.task', 'pose_landmarker_lite.task']

    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            print(f"[OK] 模型文件存在: {model}")
        else:
            print(f"[ERROR] 模型文件缺失: {model}")
            print(f"       请运行: python learning/download_models.py")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] 所有检查通过！环境配置正确。")
        print("\n下一步：运行 01_python_basics.py 学习Python基础")
    else:
        print("[FAILURE] 部分检查未通过，请根据提示修复。")
    print("=" * 60)

if __name__ == '__main__':
    main()
