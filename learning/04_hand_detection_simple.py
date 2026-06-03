"""
第4阶段：实时手部检测 - 基础手部检测

本脚本帮助您学习如何使用MediaPipe进行手部关键点检测。
"""

import cv2
import sys

def main():
    print("=" * 60)
    print("第4阶段：实时手部检测 - 基础手部检测")
    print("=" * 60)

    # 添加项目路径到sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.detection.hand_detector import HandDetector
    except ImportError as e:
        print(f"[ERROR] 无法导入HandDetector: {e}")
        print("请确保已安装所有依赖并下载模型文件")
        return

    print("\n初始化手部检测器...")

    try:
        detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        print("[OK] 手部检测器初始化成功")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # 打开摄像头
    print("\n打开摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return

    print("[OK] 摄像头打开成功")

    print("\n开始实时检测...")
    print("操作说明:")
    print("  - 按 'q' 键退出")
    print("  - 将手放在摄像头前进行检测")

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] 无法读取摄像头画面")
            break

        # 镜像翻转
        frame = cv2.flip(frame, 1)

        # 检测手部关键点
        results = detector.detect(frame)

        # 检查检测结果
        if results.hand_landmarks:
            num_hands = len(results.hand_landmarks)
            print(f"\r检测到 {num_hands} 只手", end='')

            # 显示手部数量
            cv2.putText(frame, f"Hands: {num_hands}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示关键点数量
            total_landmarks = sum(len(hand) for hand in results.hand_landmarks)
            cv2.putText(frame, f"Landmarks: {total_landmarks}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print("\r未检测到手", end='')
            cv2.putText(frame, "No hands detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示画面
        cv2.imshow('Hand Detection', frame)

        # 等待按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\n\n" + "=" * 60)
    print("基础手部检测学习完成！")
    print("下一步：运行 04_hand_detection_draw.py 学习绘制关键点")
    print("=" * 60)

if __name__ == '__main__':
    import os
    main()
