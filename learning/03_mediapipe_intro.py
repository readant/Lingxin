"""
第3阶段：MediaPipe概念入门

本脚本帮助您了解MediaPipe框架的基本概念和使用方法。
"""

import sys

def section_1_introduction():
    """3.1 MediaPipe简介"""
    print("\n" + "=" * 50)
    print("3.1 MediaPipe简介")
    print("=" * 50)

    print("""
什么是MediaPipe？
----------------
MediaPipe是Google开发的一个开源框架，用于构建多模态（音频、视频、传感器）
机器学习管道。它提供了一系列预训练模型，可以轻松实现：

1. 手部关键点检测（Hand Landmarker）
2. 人体姿态估计（Pose Landmarker）
3. 面部关键点检测（Face Landmarker）
4. 物体检测（Object Detection）
5. 手势识别（Gesture Recognition）

为什么使用MediaPipe？
-------------------
✅ 轻量级：适合移动端和边缘设备
✅ 高性能：实时处理能力强
✅ 跨平台：支持Python、C++、Android、iOS、Web
✅ 预训练模型：开箱即用

本项目使用的功能：
-----------------
- Hand Landmarker: 检测手部21个关键点
- Pose Landmarker: 检测人体姿态关键点
""")

def section_2_api_comparison():
    """3.2 API版本对比"""
    print("\n" + "=" * 50)
    print("3.2 API版本对比")
    print("=" * 50)

    print("""
旧版API (Solutions API) vs 新版API (Task API)
---------------------------------------------

┌─────────────────┬─────────────────────────┬─────────────────────────┐
│     特性        │     旧版 Solutions API   │     新版 Task API       │
├─────────────────┼─────────────────────────┼─────────────────────────┤
│ 版本要求        │ mediapipe < 0.10.33     │ mediapipe >= 0.10.33   │
│ API风格         │ 旧版风格，函数式调用     │ 现代风格，面向对象      │
│ 模型加载        │ 自动下载                │ 手动下载，本地加载      │
│ 推理速度        │ 一般                    │ ✅ 更快                 │
│ Web支持         │ 一般                    │ ✅ WebGPU支持           │
│ LLM集成         │ 不支持                  │ ✅ 支持                 │
│ 内存占用        │ 一般                    │ ✅ 更低                 │
└─────────────────┴─────────────────────────┴─────────────────────────┘

本项目使用新版Task API的原因：
----------------------------
1. 性能更好，适合实时应用
2. 支持Web部署（未来扩展）
3. API设计更现代、更清晰
4. 官方推荐使用新版API

注意事项：
--------
- 新版API需要手动下载模型文件
- API接口与旧版不兼容
- 需要学习新的使用方式
""")

def section_3_model_files():
    """3.3 模型文件说明"""
    print("\n" + "=" * 50)
    print("3.3 模型文件说明")
    print("=" * 50)

    print("""
模型文件格式：.task
------------------
MediaPipe Task API使用.tflite模型打包成.task文件。

本项目需要的模型：
-----------------
1. hand_landmarker.task    - 手部关键点检测模型
   - 大小：约10MB
   - 检测：每只手21个关键点
   - 支持：最多同时检测2只手

2. pose_landmarker_lite.task - 姿态关键点检测模型（轻量版）
   - 大小：约20MB
   - 检测：全身33个关键点（我们使用上半身15个）
   - 特点：速度快，适合实时应用

模型下载方式：
------------
方法1：运行下载脚本
   python learning/download_models.py

方法2：手动下载
   手部模型: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   姿态模型: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

模型文件存放位置：
----------------
Lingxin/
└── models/
    ├── hand_landmarker.task
    └── pose_landmarker_lite.task
""")

def section_4_key_points_intro():
    """3.4 关键点数据结构"""
    print("\n" + "=" * 50)
    print("3.4 关键点数据结构")
    print("=" * 50)

    print("""
手部关键点（21个）
-----------------
索引 | 位置        | 索引 | 位置
-----|------------|-----|------------
0    | 手腕中心    | 11  | 中指中间关节
1    | 拇指根部    | 12  | 中指指尖
2    | 拇指第一关节 | 13  | 无名指根部
3    | 拇指第二关节 | 14  | 无名指第一关节
4    | 拇指指尖    | 15  | 无名指中间关节
5    | 食指根部    | 16  | 无名指指尖
6    | 食指第一关节 | 17  | 小指根部
7    | 食指中间关节 | 18  | 小指第一关节
8    | 食指指尖    | 19  | 小指中间关节
9    | 中指根部    | 20  | 小指指尖
10   | 中指第一关节 |     |

姿态关键点（上半身15个）
-----------------------
索引 | 位置
-----|------------
0    | 鼻子
1    | 左眼内眼角
2    | 左眼外眼角
3    | 右眼内眼角
4    | 右眼外眼角
5    | 左耳
6    | 右耳
7    | 左肩
8    | 右肩
9    | 左手肘
10   | 右手肘
11   | 左手腕
12   | 右手腕
13   | 左髋
14   | 右髋

关键点坐标格式：
---------------
每个关键点包含3个坐标：(x, y, z)

- x: 水平坐标（0-1，相对于图像宽度）
- y: 垂直坐标（0-1，相对于图像高度）
- z: 深度坐标（相对于手腕，值越小越靠近摄像头）

需要转换为像素坐标：
x_pixel = x * image_width
y_pixel = y * image_height
""")

def section_5_check_mediapipe():
    """3.5 检查MediaPipe安装"""
    print("\n" + "=" * 50)
    print("3.5 检查MediaPipe安装")
    print("=" * 50)

    try:
        import mediapipe as mp
        print(f"[OK] MediaPipe 已安装，版本: {mp.__version__}")

        # 检查是否支持Task API
        if hasattr(mp, 'tasks'):
            print("[OK] 支持新版Task API")
        else:
            print("[ERROR] 不支持Task API，请升级MediaPipe")
            print("       运行: pip install --upgrade mediapipe>=0.10.33")
            return False

        # 检查模型文件
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, 'models')

        required_models = ['hand_landmarker.task', 'pose_landmarker_lite.task']
        all_found = True

        for model in required_models:
            model_path = os.path.join(models_dir, model)
            if os.path.exists(model_path):
                print(f"[OK] 模型文件存在: {model}")
            else:
                print(f"[ERROR] 模型文件缺失: {model}")
                print(f"       请运行: python learning/download_models.py")
                all_found = False

        return all_found

    except ImportError:
        print("[ERROR] MediaPipe 未安装")
        print("       运行: pip install mediapipe>=0.10.33")
        return False

def main():
    print("=" * 60)
    print("第3阶段：MediaPipe概念入门")
    print("=" * 60)

    section_1_introduction()
    section_2_api_comparison()
    section_3_model_files()
    section_4_key_points_intro()
    section_5_check_mediapipe()

    print("\n" + "=" * 60)
    print("MediaPipe概念学习完成！")
    print("下一步：运行 04_hand_detection_simple.py 学习手部检测")
    print("=" * 60)

if __name__ == '__main__':
    main()
