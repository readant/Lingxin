"""
第2阶段：OpenCV基础入门 - 图像基本操作

本脚本帮助您学习OpenCV的基础图像操作。
"""

import cv2
import numpy as np
import os

def section_1_read_image():
    """2.1 读取图像"""
    print("\n" + "=" * 50)
    print("2.1 读取图像")
    print("=" * 50)

    # 创建一个测试图像（因为可能没有test.jpg文件）
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(test_image, "OpenCV Test", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('test.jpg', test_image)

    # 读取图像
    image = cv2.imread('test.jpg')

    if image is None:
        print("[ERROR] 无法读取图像文件")
        return None

    print(f"图像形状: {image.shape}")  # (高度, 宽度, 通道数)
    print(f"图像类型: {image.dtype}")
    print(f"图像大小: {image.size} 像素")

    return image

def section_2_display_image(image):
    """2.2 显示图像"""
    print("\n" + "=" * 50)
    print("2.2 显示图像")
    print("=" * 50)

    if image is None:
        print("[ERROR] 没有图像数据")
        return

    # 显示图像
    print("按任意键关闭窗口...")
    cv2.imshow('My Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def section_3_save_image(image):
    """2.3 保存图像"""
    print("\n" + "=" * 50)
    print("2.3 保存图像")
    print("=" * 50)

    if image is None:
        print("[ERROR] 没有图像数据")
        return

    # 保存图像
    cv2.imwrite('output.jpg', image)
    print("[OK] 图像已保存为 output.jpg")

    # 检查文件是否存在
    if os.path.exists('output.jpg'):
        print("[OK] 文件保存成功")
    else:
        print("[ERROR] 文件保存失败")

def section_4_image_operations(image):
    """2.4 图像基本操作"""
    print("\n" + "=" * 50)
    print("2.4 图像基本操作")
    print("=" * 50)

    if image is None:
        print("[ERROR] 没有图像数据")
        return

    # 1. 访问像素
    pixel = image[50, 50]
    print(f"像素值 (B, G, R): {pixel}")

    # 2. 修改像素
    image[50, 50] = [0, 0, 255]  # 设置为红色
    print("[OK] 已修改像素值")

    # 3. 裁剪图像
    cropped = image[50:150, 50:250]
    print(f"裁剪后图像形状: {cropped.shape}")

    # 4. 调整大小
    resized = cv2.resize(image, (150, 100))
    print(f"调整大小后: {resized.shape}")

    # 5. 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"灰度图像形状: {gray.shape}")

    # 显示处理后的图像
    print("\n显示处理后的图像（按任意键关闭）...")
    cv2.imshow('Cropped', cropped)
    cv2.imshow('Resized', resized)
    cv2.imshow('Grayscale', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def section_5_drawing(image):
    """2.5 在图像上绘图"""
    print("\n" + "=" * 50)
    print("2.5 在图像上绘图")
    print("=" * 50)

    if image is None:
        print("[ERROR] 没有图像数据")
        return

    # 创建图像副本
    drawing = image.copy()

    # 绘制直线
    cv2.line(drawing, (50, 50), (250, 150), (0, 255, 0), 2)

    # 绘制矩形
    cv2.rectangle(drawing, (50, 50), (150, 150), (0, 0, 255), 2)

    # 绘制圆形
    cv2.circle(drawing, (200, 100), 30, (255, 0, 0), -1)  # -1表示填充

    # 添加文字
    cv2.putText(drawing, "Hello OpenCV!", (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 显示结果
    print("显示绘图结果（按任意键关闭）...")
    cv2.imshow('Drawing', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("=" * 60)
    print("第2阶段：OpenCV基础入门 - 图像基本操作")
    print("=" * 60)

    # 读取图像
    image = section_1_read_image()

    # 显示图像
    section_2_display_image(image)

    # 保存图像
    section_3_save_image(image)

    # 图像操作
    section_4_image_operations(image)

    # 绘图
    section_5_drawing(image)

    # 清理测试文件
    for f in ['test.jpg', 'output.jpg']:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 60)
    print("OpenCV图像基础操作学习完成！")
    print("下一步：运行 02_opencv_video.py 学习视频处理")
    print("=" * 60)

if __name__ == '__main__':
    main()
