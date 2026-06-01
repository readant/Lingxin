"""
第10阶段：项目实战 - 数据采集

本脚本实现完整的数据采集流程，用于收集手语数据。
"""

import cv2
import numpy as np
import os
import sys

def main():
    print("=" * 60)
    print("第10阶段：项目实战 - 数据采集")
    print("=" * 60)
    
    # 添加项目路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    
    try:
        from src.detection.hand_detector import HolisticDetector
    except ImportError as e:
        print(f"[ERROR] 无法导入HolisticDetector: {e}")
        return
    
    print("\n[1] 初始化配置")
    print("-" * 40)
    
    # 配置参数
    person_id = input("请输入人员ID（如user001）: ").strip() or "user001"
    vocab_id = input("请输入词汇编号（如01）: ").strip() or "01"
    record_duration = int(input("录制时长（秒，默认3秒）: ") or "3")
    
    print(f"\n配置信息:")
    print(f"  人员ID: {person_id}")
    print(f"  词汇编号: {vocab_id}")
    print(f"  录制时长: {record_duration}秒")
    
    # 创建数据目录
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"  数据保存目录: {data_dir}")
    
    print("\n[2] 初始化检测器")
    print("-" * 40)
    
    try:
        detector = HolisticDetector(
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        print("[OK] HolisticDetector初始化成功")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    print("\n[3] 打开摄像头")
    print("-" * 40)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    print("[OK] 摄像头打开成功")
    
    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  分辨率: {width} x {height}")
    print(f"  帧率: {fps:.1f} FPS")
    
    print("\n[4] 数据采集")
    print("-" * 40)
    print("操作说明:")
    print("  - 按 'r' 键开始录制")
    print("  - 录制期间会显示红框")
    print("  - 按 'q' 键退出")
    
    recording = False
    recorded_frames = []
    frame_count = 0
    start_time = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] 无法读取摄像头画面")
            break
        
        # 镜像翻转
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # 检测关键点
        results = detector.detect(frame)
        
        # 绘制关键点
        frame = detector.draw_landmarks(frame, results)
        
        # 录制逻辑
        if recording:
            # 绘制录制边框
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)
            cv2.putText(frame, "RECORDING", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 获取关键点数据
            landmarks = detector.get_landmarks(results, frame.shape)
            recorded_frames.append(landmarks)
            
            # 计算录制进度
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            progress = min(elapsed / record_duration, 1) * 100
            
            cv2.putText(frame, f"进度: {progress:.1f}%", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 检查是否录制完成
            if elapsed >= record_duration:
                print(f"\n[OK] 录制完成！共 {len(recorded_frames)} 帧")
                recording = False
                
                # 保存数据
                data = np.array(recorded_frames)
                filename = f"{person_id}_{vocab_id}.npy"
                filepath = os.path.join(data_dir, filename)
                np.save(filepath, data)
                print(f"[OK] 数据已保存: {filepath}")
                print(f"  数据形状: {data.shape}")
                
                recorded_frames = []
        else:
            cv2.putText(frame, "Press 'R' to Record", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示帧率
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow('Data Collection', frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r') and not recording:
            print("\n[START] 开始录制...")
            recording = True
            start_time = cv2.getTickCount()
            recorded_frames = []
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("\n" + "=" * 60)
    print("数据采集完成！")
    print("下一步：运行 10_model_training.py 训练模型")
    print("=" * 60)

if __name__ == '__main__':
    main()