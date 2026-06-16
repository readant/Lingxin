"""
第8阶段：机器学习基础 - SVM分类器原理

本脚本帮助您学习支持向量机（SVM）的基本原理和使用方法。
"""

import numpy as np
import matplotlib.pyplot as plt

def section_1_classification_intro():
    """8.1 分类问题介绍"""
    print("\n" + "=" * 50)
    print("8.1 分类问题介绍")
    print("=" * 50)

    print("""
什么是分类问题？
---------------
分类是机器学习中的监督学习任务，目标是将输入数据分配到预定义的类别中。

例如：
- 垃圾邮件检测：垃圾邮件 vs 正常邮件
- 图像识别：猫 vs 狗
- 手语识别：手势A vs 手势B vs ...

分类问题类型：
-------------
1. 二分类：只有两个类别（是/否）
2. 多分类：多个类别

评估指标：
---------
- 准确率（Accuracy）：正确预测的比例
- 精确率（Precision）：预测为正例的样本中真正为正例的比例
- 召回率（Recall）：真正为正例的样本中被正确预测的比例
- F1分数：精确率和召回率的调和平均数
""")

def section_2_svm_principle():
    """8.2 SVM原理"""
    print("\n" + "=" * 50)
    print("8.2 SVM原理")
    print("=" * 50)

    print("""
什么是SVM？
----------
支持向量机（Support Vector Machine）是一种强大的分类算法。

核心思想：
---------
寻找一个最优超平面，将不同类别的数据分开，并且最大化分类间隔。

关键概念：
---------
1. 超平面：在二维空间是一条直线，在三维空间是一个平面，在高维空间是超平面
2. 支持向量：距离超平面最近的样本点
3. 分类间隔：两个类别支持向量之间的距离
4. 核函数：将低维数据映射到高维空间，解决非线性问题

SVM的优点：
----------
- 在高维空间中表现良好
- 泛化能力强，不容易过拟合
- 适用于小样本数据集

常用核函数：
-----------
1. 线性核：适用于线性可分数据
2. 多项式核：适用于多项式关系数据
3. RBF（径向基函数）核：适用于非线性数据

SVM的应用场景：
--------------
- 图像分类
- 文本分类
- 生物信息学
- 手语识别
""")

def section_3_svm_practice():
    """8.3 使用scikit-learn实现SVM"""
    print("\n" + "=" * 50)
    print("8.3 使用scikit-learn实现SVM")
    print("=" * 50)

    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # 生成模拟数据
    print("步骤1: 生成模拟数据")
    X, y = make_classification(
        n_samples=100,     # 样本数量
        n_features=20,     # 特征数量
        n_classes=5,       # 类别数量
        n_informative=15,  # 有效特征数量
        random_state=42
    )
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 划分训练集和测试集
    print("\n步骤2: 划分训练集和测试集")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 创建SVM分类器
    print("\n步骤3: 创建SVM分类器")
    svm = SVC(
        kernel='rbf',       # 使用RBF核
        C=1.0,              # 正则化参数
        gamma='scale',      # 核函数参数
        random_state=42
    )
    print(f"SVM参数: {svm.get_params()}")

    # 训练模型
    print("\n步骤4: 训练模型")
    svm.fit(X_train, y_train)

    # 预测
    print("\n步骤5: 预测")
    y_pred = svm.predict(X_test)

    # 评估
    print("\n步骤6: 评估模型")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.2f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

def section_4_hand_sign_example():
    """8.4 手语识别示例"""
    print("\n" + "=" * 50)
    print("8.4 手语识别示例")
    print("=" * 50)

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 模拟手语数据
    print("模拟手语数据:")
    print("- 100个样本")
    print("- 每个样本71维特征（相对坐标63 + 手指长度4 + 关节角度4）")
    print("- 5个手势类别")

    np.random.seed(42)
    n_samples = 100
    n_features = 71  # 项目中SVM/RF/MLP使用71维提取特征
    n_classes = 5

    # 生成模拟数据（每个类别有不同的特征模式）
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_classes):
        start = i * (n_samples // n_classes)
        end = (i + 1) * (n_samples // n_classes)
        X[start:end] = np.random.randn(end - start, n_features) + i * 0.5
        y[start:end] = i

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 创建并训练SVM
    svm = SVC(kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train, y_train)

    # 预测和评估
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n训练完成！")
    print(f"训练集准确率: {svm.score(X_train, y_train):.2f}")
    print(f"测试集准确率: {accuracy:.2f}")

    # 特征重要性分析（SVM的方式）
    print("\n支持向量数量:")
    print(f"  总支持向量: {len(svm.support_vectors_)}")
    print(f"  各类别支持向量: {svm.n_support_}")

def main():
    print("=" * 60)
    print("第8阶段：机器学习基础 - SVM分类器原理")
    print("=" * 60)

    section_1_classification_intro()
    section_2_svm_principle()
    section_3_svm_practice()
    section_4_hand_sign_example()

    print("\n" + "=" * 60)
    print("SVM学习完成！")
    print("下一步：运行 09_lstm_intro.py 学习深度学习")
    print("=" * 60)

if __name__ == '__main__':
    main()
