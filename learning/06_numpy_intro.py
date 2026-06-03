"""
第6阶段：NumPy数据处理 - NumPy基础入门

本脚本帮助您学习NumPy数组的基本操作。
"""

import numpy as np

def section_1_array_creation():
    """6.1 创建NumPy数组"""
    print("\n" + "=" * 50)
    print("6.1 创建NumPy数组")
    print("=" * 50)

    # 从列表创建
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"从列表创建: arr1 = {arr1}")
    print(f"类型: {type(arr1)}")
    print(f"形状: {arr1.shape}")

    # 创建二维数组
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\n二维数组: \n{arr2}")
    print(f"形状: {arr2.shape}")

    # 特殊数组
    print("\n特殊数组:")
    print(f"全零数组: {np.zeros(5)}")
    print(f"全一数组: {np.ones(5)}")
    print(f"单位矩阵:\n{np.eye(3)}")
    print(f"随机数组:\n{np.random.rand(2, 3)}")

    # 数组范围
    print(f"\n数组范围:")
    print(f"np.arange(10): {np.arange(10)}")
    print(f"np.linspace(0, 1, 5): {np.linspace(0, 1, 5)}")

def section_2_array_indexing():
    """6.2 数组索引和切片"""
    print("\n" + "=" * 50)
    print("6.2 数组索引和切片")
    print("=" * 50)

    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(f"原始数组: {arr}")

    # 访问单个元素
    print(f"\n访问单个元素:")
    print(f"arr[0] = {arr[0]}")
    print(f"arr[-1] = {arr[-1]}")

    # 切片
    print(f"\n切片操作:")
    print(f"arr[1:5] = {arr[1:5]}")
    print(f"arr[:5] = {arr[:5]}")
    print(f"arr[5:] = {arr[5:]}")
    print(f"arr[::2] = {arr[::2]}")  # 步长为2

    # 二维数组索引
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\n二维数组:\n{arr2d}")
    print(f"arr2d[0, 0] = {arr2d[0, 0]}")
    print(f"arr2d[1, :] = {arr2d[1, :]}")
    print(f"arr2d[:, 1] = {arr2d[:, 1]}")

def section_3_array_operations():
    """6.3 数组运算"""
    print("\n" + "=" * 50)
    print("6.3 数组运算")
    print("=" * 50)

    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([6, 7, 8, 9, 10])

    print(f"arr1 = {arr1}")
    print(f"arr2 = {arr2}")

    # 基本运算
    print(f"\n基本运算:")
    print(f"arr1 + arr2 = {arr1 + arr2}")
    print(f"arr1 * arr2 = {arr1 * arr2}")
    print(f"arr1 ** 2 = {arr1 ** 2}")
    print(f"np.sin(arr1) = {np.sin(arr1)}")

    # 矩阵运算
    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    print(f"\n矩阵运算:")
    print(f"mat1:\n{mat1}")
    print(f"mat2:\n{mat2}")
    print(f"mat1 @ mat2 (矩阵乘法):\n{mat1 @ mat2}")

    # 统计运算
    print(f"\n统计运算:")
    print(f"arr1.sum() = {arr1.sum()}")
    print(f"arr1.mean() = {arr1.mean()}")
    print(f"arr1.max() = {arr1.max()}")
    print(f"arr1.min() = {arr1.min()}")
    print(f"arr1.std() = {arr1.std()}")

def section_4_array_shape():
    """6.4 数组形状操作"""
    print("\n" + "=" * 50)
    print("6.4 数组形状操作")
    print("=" * 50)

    arr = np.arange(12)
    print(f"原始数组: {arr}")
    print(f"形状: {arr.shape}")

    # 改变形状
    arr_2d = arr.reshape(3, 4)
    print(f"\nreshape(3, 4):\n{arr_2d}")

    # 扁平化
    arr_flat = arr_2d.flatten()
    print(f"\nflatten(): {arr_flat}")

    # 转置
    arr_trans = arr_2d.T
    print(f"\n转置:\n{arr_trans}")

    # 拼接
    arr_a = np.array([[1, 2], [3, 4]])
    arr_b = np.array([[5, 6], [7, 8]])
    print(f"\n垂直拼接:\n{np.vstack([arr_a, arr_b])}")
    print(f"水平拼接:\n{np.hstack([arr_a, arr_b])}")

def section_5_practical_example():
    """6.5 实际应用示例 - 关键点数据处理"""
    print("\n" + "=" * 50)
    print("6.5 实际应用示例 - 关键点数据处理")
    print("=" * 50)

    # 模拟手部关键点数据 (21个关键点，每个点3个坐标)
    hand_landmarks = np.random.rand(21, 3) * 100  # 模拟像素坐标
    print(f"手部关键点形状: {hand_landmarks.shape}")
    print(f"手部关键点数据:\n{hand_landmarks[:3]}...")  # 只显示前3个

    # 计算相对坐标（相对于手腕）
    wrist = hand_landmarks[0]  # 手腕关键点
    relative_landmarks = hand_landmarks - wrist
    print(f"\n相对坐标（相对于手腕）:\n{relative_landmarks[:3]}...")

    # 归一化
    max_val = np.max(np.abs(relative_landmarks))
    normalized_landmarks = relative_landmarks / max_val
    print(f"\n归一化后:\n{normalized_landmarks[:3]}...")

    # 计算手指长度
    thumb_length = np.linalg.norm(relative_landmarks[4] - relative_landmarks[0])
    index_length = np.linalg.norm(relative_landmarks[8] - relative_landmarks[5])
    print(f"\n手指长度:")
    print(f"拇指长度: {thumb_length:.2f}")
    print(f"食指长度: {index_length:.2f}")

    # 转换为一维特征向量
    feature_vector = hand_landmarks.flatten()
    print(f"\n一维特征向量形状: {feature_vector.shape}")

def main():
    print("=" * 60)
    print("第6阶段：NumPy数据处理 - NumPy基础入门")
    print("=" * 60)

    section_1_array_creation()
    section_2_array_indexing()
    section_3_array_operations()
    section_4_array_shape()
    section_5_practical_example()

    print("\n" + "=" * 60)
    print("NumPy基础学习完成！")
    print("下一步：运行 06_numpy_operations.py 学习NumPy进阶操作")
    print("=" * 60)

if __name__ == '__main__':
    main()
