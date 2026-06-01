"""
第2阶段：OpenCV基础入门 - 视频处理

本脚本帮助您学习如何访问摄像头和处理视频流。
"""

import cv2
import sys

def section_1_access_camera():
    """2.1 访问摄像头"""
    print("\n" + "=" * 50)
    print("2.1 访问摄像头")
    print("=" * 50)
    
    # 打开摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否打开成功
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        print("请检查：")
        print("1. 摄像头是否被其他程序占用")
        print("2. 摄像头驱动是否安装")
        print("3. 尝试更换摄像头索引 (1, 2, ...)")
        return
    
    print("[OK] 摄像头打开成功")
    
    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"摄像头参数:")
    print(f"  分辨率: {width} x {height}")
    print(f"  帧率: {fps} FPS")
    
    return cap

def section_2_display_video(cap):
    """2.2 显示实时视频"""
    print("\n" + "=" * 50)
    print("2.2 显示实时视频")
    print("=" * 50)
    
    if cap is None:
        print("[ERROR] 没有摄像头对象")
        return
    
    print("显示摄像头画面...")
    print("操作说明:")
    print("  - 按 'q' 键退出")
    print("  - 按 's' 键保存当前帧")
    print("  - 按 'm' 键切换镜像模式")
    
    mirror_mode = True
    frame_count = 0
    saved_count = 0
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] 无法读取摄像头画面")
            break
        
        frame_count += 1
        
        # 镜像翻转
        if mirror_mode:
            frame = cv2.flip(frame, 1)
        
        # 在画面上显示信息
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mirror: {'ON' if mirror_mode else 'OFF'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示画面
        cv2.imshow('Camera', frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # 退出
            break
        elif key == ord('s'):
            # 保存当前帧
            filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"[OK] 帧已保存: {filename}")
        elif key == ord('m'):
            # 切换镜像模式
            mirror_mode = not mirror_mode
            print(f"[OK] 镜像模式: {'ON' if mirror_mode else 'OFF'}")
    
    print(f"\n共读取 {frame_count} 帧，保存 {saved_count} 帧")

def section_3_video_writer():
    """2.3 录制视频"""
    print("\n" + "=" * 50)
    print("2.3 录制视频")
    print("=" * 50)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头")
        return
    
    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0  # 设置录制帧率
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    print("开始录制视频...")
    print("操作说明:")
    print("  - 按 'q' 键停止录制")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 镜像翻转
        frame = cv2.flip(frame, 1)
        
        # 写入视频
        out.write(frame)
        
        # 显示录制状态
        cv2.putText(frame, "RECORDING", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Recording', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("[OK] 视频已保存为 output.mp4")

def section_4_video_playback():
    """2.4 播放视频文件"""
    print("\n" + "=" * 50)
    print("2.4 播放视频文件")
    print("=" * 50)
    
    # 检查视频文件是否存在
    if not cv2.os.path.exists('output.mp4'):
        print("[ERROR] 视频文件不存在，请先录制视频")
        return
    
    # 打开视频文件
    cap = cv2.VideoCapture('output.mp4')
    
    if not cap.isOpened():
        print("[ERROR] 无法打开视频文件")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频信息:")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps} FPS")
    print(f"  时长: {total_frames / fps:.2f} 秒")
    
    print("\n播放视频...")
    print("操作说明:")
    print("  - 按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # 视频播放完毕
            break
        
        cv2.imshow('Video Playback', frame)
        
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=" * 60)
    print("第2阶段：OpenCV基础入门 - 视频处理")
    print("=" * 60)
    
    # 访问摄像头
    cap = section_1_access_camera()
    
    # 显示实时视频
    if cap:
        section_2_display_video(cap)
        cap.release()
        cv2.destroyAllWindows()
    
    # 录制视频
    section_3_video_writer()
    
    # 播放视频
    section_4_video_playback()
    
    print("\n" + "=" * 60)
    print("OpenCV视频处理学习完成！")
    print("下一步：运行 03_mediapipe_intro.py 学习MediaPipe")
    print("=" * 60)

if __name__ == '__main__':
    main()