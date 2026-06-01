"""
第6阶段：NumPy数据处理 - NumPy进阶操作

本脚本帮助您学习NumPy的高级操作和数据处理技巧。
"""

import numpy as np

def section_1_broadcasting():
    """6.1 广播机制"""
    print("\n" + "=" * 50)
    print("6.1 广播机制")
    print("=" * 50)
    
    print("""
广播机制允许NumPy在不同形状的数组之间进行运算。

规则：
1. 如果两个数组的维度数不同，将维度较小的数组进行扩展
2. 如果在某个维度上的大小相同，或者其中一个数组在该维度上的大小为1，则可以广播
""")
    
    # 示例1：标量与数组
    arr = np.array([1, 2, 3, 4, 5])
    result = arr + 10
    print(f"示例1: arr + 10")
    print(f"arr = {arr}")
    print(f"result = {result}")
    
    # 示例2：一维数组与二维数组
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr1d = np.array([10, 20, 30])
    result = arr2d + arr1d
    print(f"\n示例2: arr2d + arr1d")
    print(f"arr2d:\n{arr2d}")
    print(f"arr1d = {arr1d}")
    print(f"result:\n{result}")
    
    # 示例3：不同维度数组
    arr3 = np.array([1, 2, 3])
    arr3 = arr3.reshape(3, 1)
    result = arr2d * arr3
    print(f"\n示例3: arr2d * arr3 (arr3形状: {arr3.shape})")
    print(f"arr3:\n{arr3}")
    print(f"result:\n{result}")

def section_2_masking_filtering():
    """6.2 掩码和过滤"""
    print("\n" + "=" * 50)
    print("6.2 掩码和过滤")
    print("=" * 50)
    
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"原始数组: {arr}")
    
    # 创建掩码
    mask = arr > 5
    print(f"\n掩码 (arr > 5): {mask}")
    
    # 使用掩码过滤
    filtered = arr[mask]
    print(f"过滤结果: {filtered}")
    
    # 直接过滤
    result = arr[arr % 2 == 0]
    print(f"\n偶数元素: {result}")
    
    # 修改满足条件的元素
    arr[arr > 5] = 0
    print(f"\n将大于5的元素设为0: {arr}")

def section_3_advanced_indexing():
    """6.3 高级索引"""
    print("\n" + "=" * 50)
    print("6.3 高级索引")
    print("=" * 50)
    
    arr = np.arange(12).reshape(3, 4)
    print(f"原始数组:\n{arr}")
    
    # 整数数组索引
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 3])
    print(f"\n整数数组索引: arr[rows, cols]")
    print(f"rows = {rows}")
    print(f"cols = {cols}")
    print(f"结果: {arr[rows, cols]}")
    
    # 布尔索引
    mask = arr > 5
    print(f"\n布尔索引: arr[mask]")
    print(f"掩码:\n{mask}")
    print(f"结果: {arr[mask]}")
    
    # 结合使用
    print(f"\n提取第二行中大于5的元素: {arr[1, arr[1] > 5]}")

def section_4_performance_tips():
    """6.4 性能优化技巧"""
    print("\n" + "=" * 50)
    print("6.4 性能优化技巧")
    print("=" * 50)
    
    print("""
NumPy性能优化要点：
------------------
1. 使用向量化操作，避免Python循环
2. 使用内置函数，它们是用C实现的
3. 避免不必要的数组拷贝
4. 使用适当的数据类型

性能对比：
---------
""")
    
    # 性能对比示例
    import time
    
    # 方法1：Python循环
    arr = np.arange(1000000)
    start = time.time()
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] * 2 + 1
    python_time = time.time() - start
    
    # 方法2：NumPy向量化
    start = time.time()
    result = arr * 2 + 1
    numpy_time = time.time() - start
    
    print(f"Python循环时间: {python_time:.4f} 秒")
    print(f"NumPy向量化时间: {numpy_time:.4f} 秒")
    print(f"加速比: {python_time / numpy_time:.1f}x")

def section_5_data_normalization():
    """6.5 数据标准化和归一化"""
    print("\n" + "=" * 50)
    print("6.5 数据标准化和归一化")
    print("=" * 50)
    
    # 创建模拟数据
    data = np.random.rand(100, 5) * 100  # 100个样本，5个特征
    print(f"数据形状: {data.shape}")
    print(f"原始数据统计:")
    print(f"  均值: {data.mean(axis=0)}")
    print(f"  标准差: {data.std(axis=0)}")
    print(f"  最小值: {data.min(axis=0)}")
    print(f"  最大值: {data.max(axis=0)}")
    
    # 方法1：min-max归一化 (缩放到[0, 1])
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    print(f"\nMin-Max归一化后:")
    print(f"  最小值: {normalized_data.min(axis=0)}")
    print(f"  最大值: {normalized_data.max(axis=0)}")
    
    # 方法2：Z-score标准化 (均值为0，标准差为1)
    mean_val = data.mean(axis=0)
    std_val = data.std(axis=0)
    standardized_data = (data - mean_val) / std_val
    print(f"\nZ-score标准化后:")
    print(f"  均值: {standardized_data.mean(axis=0).round(4)}")
    print(f"  标准差: {standardized_data.std(axis=0).round(4)}")

def section_6_practical_example():
    """6.6 实际应用示例 - 时序数据处理"""
    print("\n" + "=" * 50)
    print("6.6 实际应用示例 - 时序数据处理")
    print("=" * 50)
    
    # 模拟时序数据：30帧，每帧171维特征
    timesteps = 30
    features = 171
    sequence_data = np.random.rand(timesteps, features)
    
    print(f"时序数据形状: {sequence_data.shape}")
    print(f"含义: {timesteps}帧 × {features}维特征")
    
    # 计算每帧的均值
    frame_means = sequence_data.mean(axis=1)
    print(f"\n每帧均值 (形状: {frame_means.shape}):")
    print(f"前5帧: {frame_means[:5].round(4)}")
    
    # 计算每个特征在时间上的变化
    time_diff = np.diff(sequence_data, axis=0)
    print(f"\n时间差分 (形状: {time_diff.shape}):")
    print(f"第一帧差分的前5个特征: {time_diff[0, :5].round(4)}")
    
    # 归一化整个序列
    mean = sequence_data.mean()
    std = sequence_data.std()
    normalized_sequence = (sequence_data - mean) / std
    print(f"\n序列归一化后:")
    print(f"  整体均值: {normalized_sequence.mean().round(4)}")
    print(f"  整体标准差: {normalized_sequence.std().round(4)}")

def main():
    print("=" * 60)
    print("第6阶段：NumPy数据处理 - NumPy进阶操作")
    print("=" * 60)
    
    section_1_broadcasting()
    section_2_masking_filtering()
    section_3_advanced_indexing()
    section_4_performance_tips()
    section_5_data_normalization()
    section_6_practical_example()
    
    print("\n" + "=" * 60)
    print("NumPy进阶学习完成！")
    print("下一步：运行 07_feature_extraction.py 学习特征工程")
    print("=" * 60)

if __name__ == '__main__':
    main()