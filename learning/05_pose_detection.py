"""
第5阶段：实时姿态检测

本脚本帮助您学习如何使用MediaPipe进行人体姿态关键点检测。
"""

import cv2
import sys
import os

def main():
    print("=" * 60)
    print("第5阶段：实时姿态检测")
    print("=" * 60)

    # 添加项目路径到sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)

    try:
        from src.detection.hand_detector import PoseDetector
    except ImportError as e:
        print(f"[ERROR] 无法导入PoseDetector: {e}")
        return

    print("\n初始化姿态检测器...")

    try:
        detector = PoseDetector(
            min_detection_confidence=0.5
        )
        print("[OK] 姿态检测器初始化成功")
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
    print("  - 按 'd' 键切换绘制模式")
    print("  - 站在摄像头前进行检测")

    draw_mode = True

    while True:
        # 读取一帧
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] 无法读取摄像头画面")
            break

        # 镜像翻转
        frame = cv2.flip(frame, 1)

        # 检测姿态关键点
        results = detector.detect(frame)

        # 绘制关键点和骨架
        if draw_mode:
            frame = detector.draw_landmarks(frame, results)

        # 检查检测结果
        if results.pose_landmarks:
            num_landmarks = len(results.pose_landmarks[0])
            print(f"\r检测到 {num_landmarks} 个姿态关键点", end='')
            cv2.putText(frame, f"Pose Landmarks: {num_landmarks}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print("\r未检测到人体姿态", end='')
            cv2.putText(frame, "No pose detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示模式信息
        mode_text = "Drawing: ON" if draw_mode else "Drawing: OFF"
        cv2.putText(frame, mode_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示画面
        cv2.imshow('Pose Detection', frame)

        # 等待按键
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            draw_mode = not draw_mode
            print(f"\n[OK] 绘制模式: {'ON' if draw_mode else 'OFF'}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\n" + "=" * 60)
    print("姿态检测学习完成！")
    print("下一步：运行 06_numpy_intro.py 学习NumPy")
    print("=" * 60)

if __name__ == '__main__':
    main()
