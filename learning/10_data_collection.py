"""
第10阶段：项目实战 - 数据采集

本脚本演示项目中的数据采集流程，并引导使用完整的采集工具。

项目提供两种采集方式：
1. tools/collect_data.py — 摄像头实时采集（推荐）
2. tools/collect_from_video.py — 视频文件批量采集
"""

import cv2
import numpy as np
import os
import sys

def show_project_tools():
    """展示项目已有的采集工具"""
    print("\n[项目采集工具]")
    print("-" * 40)
    print("""
项目已经实现了完整的数据采集工具，功能包括：
- 词汇表管理（50个常用词汇）
- 人员ID管理
- 空格键录制控制
- 实时质量检查
- 中文界面显示
- 视频批量采集

使用方法：
  # 摄像头实时采集
  python tools/collect_data.py

  # 视频批量采集
  python tools/collect_from_video.py -i data/videos/

数据保存格式：
  data/raw/collected/{词汇名}/{人员ID}_{序号}.npy
  data/raw/collected/{词汇名}/{人员ID}_{序号}_meta.json
""")

def demo_simple_collection():
    """演示简化版采集流程（用于理解原理）"""
    print("\n[简化版采集演示]")
    print("-" * 40)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.detection.hand_detector import HolisticDetector
    except ImportError as e:
        print(f"[ERROR] 无法导入HolisticDetector: {e}")
        return

    person_id = input("请输入人员ID（如user001）: ").strip() or "user001"
    word = input("请输入词汇（如你好）: ").strip() or "你好"

    print(f"\n配置: 人员={person_id}, 词汇={word}")
    print("注意: 这是简化版演示，正式采集请使用 tools/collect_data.py")

    try:
        detector = HolisticDetector(max_num_hands=2, min_detection_confidence=0.5)
        print("[OK] HolisticDetector初始化成功")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("\n操作: 按 'r' 开始录制, 按 'q' 退出")

    recording = False
    recorded_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = detector.detect(frame)
        frame = detector.draw_landmarks(frame, results)

        if recording:
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 5)
            landmarks = detector.get_landmarks(results, frame.shape)
            recorded_frames.append(landmarks)
            cv2.putText(frame, f"Recording: {len(recorded_frames)} frames",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 'R' to Record",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Data Collection Demo', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r') and not recording:
            recording = True
            recorded_frames = []
            print("\n开始录制...")
        elif key == ord('r') and recording:
            recording = False
            if recorded_frames:
                data = np.array(recorded_frames)
                save_dir = os.path.join(project_root, 'data', 'raw', 'collected', word)
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, f'{person_id}_demo.npy')
                np.save(filepath, data)
                print(f"录制完成! {len(recorded_frames)} 帧, 保存至: {filepath}")
                print(f"数据形状: {data.shape}")
            recorded_frames = []

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

def main():
    print("=" * 60)
    print("第10阶段：项目实战 - 数据采集")
    print("=" * 60)

    show_project_tools()

    print("\n是否运行简化版采集演示？")
    choice = input("(y/n, 默认n): ").strip().lower()
    if choice == 'y':
        demo_simple_collection()

    print("\n" + "=" * 60)
    print("数据采集学习完成！")
    print("正式采集请运行: python tools/collect_data.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
