# Lecture 2: Regularization and Optimization - 详细扩充版

## 课程概述

本节课深入探讨了深度学习和计算机视觉中的两个核心概念：**正则化（Regularization）\**和\**优化（Optimization）**。课程首先回顾了图像分类任务和线性分类器的基本原理，然后详细讲解了如何通过损失函数评估模型性能，如何使用正则化防止过拟合，以及如何通过各种优化算法（特别是梯度下降及其变体）来训练模型。

> **为什么这两个概念如此重要？**
>
> - **正则化**是防止模型过拟合的关键技术，确保模型不仅在训练数据上表现良好，更重要的是在未见过的测试数据上也能保持良好性能
> - **优化**是实际训练深度学习模型的核心方法，决定了我们能否找到好的模型参数，以及找到这些参数需要多长时间

## 目录

1. [回顾：图像分类与线性分类器](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#1-回顾图像分类与线性分类器)
2. [损失函数](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#2-损失函数)
3. [正则化（Regularization）](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#3-正则化regularization)
4. [优化（Optimization）](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#4-优化optimization)
5. [梯度下降及其变体](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#5-梯度下降及其变体)
6. [学习率调度](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#6-学习率调度)
7. [总结与展望](../../../chat/55adccd6-febc-4b35-aa71-3652736ec25e#7-总结与展望)

------

## 1. 回顾：图像分类与线性分类器

### 1.1 图像分类任务

**图像分类**是计算机视觉的核心任务之一，目标是将输入图像映射到预定义的类别标签集合中。

**任务定义**：

- **输入**：图像（像素值的多维数组）
- **输出**：类别标签（如：猫、狗、鸟、鹿、卡车）

**为什么图像分类是基础任务？**

图像分类是许多高级计算机视觉任务的基础。一旦我们能够准确地对图像进行分类，就可以将这个能力扩展到：

- **目标检测**：不仅识别物体类别，还要定位物体位置
- **语义分割**：对图像中每个像素进行分类
- **图像描述**：生成描述图像内容的自然语言
- **视觉问答**：回答关于图像内容的问题

**实际应用场景**：

- 医疗影像诊断（如X光片分类：正常/异常）
- 自动驾驶（识别行人、车辆、交通标志）
- 内容审核（识别不当内容）
- 零售业（商品识别和库存管理）
- 安防监控（人脸识别、异常行为检测）

### 1.2 图像分类的挑战

图像分类看似简单，但实际上面临许多技术挑战。这些挑战源于计算机对图像的表示方式与人类感知之间的根本差异。

1. **语义鸿沟（Semantic Gap）**

   这是计算机视觉中最根本的挑战之一。

   - **人类感知**：我们看到一张图片，立即能识别出"这是一只猫"。我们的大脑自动进行高级语义理解。
   - **计算机表示**：计算机只能"看到"像素值的数值网格，每个像素由红、绿、蓝三个通道的强度值（通常0-255）表示。

   **举例说明**：

   - 一张32×32的彩色猫图像在计算机中表示为一个3072维的向量（32×32×3）
   - 这个向量可能是：[245, 244, 243, 200, 199, 198, ...]
   - 从这些数字中直接推断出"猫"的概念是极其困难的

   **为什么这是挑战**：

   - 两只完全不同姿势的猫，它们的像素值可能相差巨大
   - 一只猫和一只狗在某些情况下的像素值可能很接近
   - 我们需要学习一种映射，能够从低级像素特征提取高级语义信息

2. **图像变化**

   同一物体在不同条件下可能呈现出完全不同的视觉外观：

   - 光照变化（Illumination）
     - 同一只猫在明亮阳光下和昏暗室内的像素值完全不同
     - 强光可能导致过曝，阴影可能隐藏细节
     - **技术挑战**：算法需要对光照变化保持不变性
   - 遮挡（Occlusion）
     - 物体可能部分被其他物体遮挡
     - 例如：猫躲在沙发靠垫后面，只露出尾巴
     - **技术挑战**：需要从局部信息推断整体
   - 形变（Deformation）
     - 非刚性物体（如猫、人）可以改变形状
     - 猫可以蜷缩、伸展、跳跃，每种姿态看起来都不同
     - **技术挑战**：需要学习形状的可变性
   - 背景杂波（Background Clutter）
     - 物体可能与背景颜色相似
     - 例如：白猫在白色床单上
     - **技术挑战**：需要区分前景和背景
   - 类内变化（Intra-class Variation）
     - 同一类别的不同实例差异可能很大
     - 波斯猫、暹罗猫、橘猫外观差异显著，但都属于"猫"类
     - **技术挑战**：需要学习类别的共同特征，同时容忍内部变化

![00:00:56](../../../assets/lecture-3/screenshot-01KG66KKM3RNW3T52AGRKH04YJ.png) [00:00:56](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**为什么简单的规则无法解决问题**：

由于上述挑战的存在，我们无法简单地编写if-else规则来进行图像分类。例如：

```python
# 这种方法是行不通的
def classify_cat(image):
    if image[100, 100] == [255, 200, 180]:  # 某个特定位置的颜色
        return "cat"
    else:
        return "not cat"
```

这就是为什么我们需要**机器学习**和**数据驱动**的方法。

### 1.3 K近邻分类器（K-Nearest Neighbors）

K近邻（KNN）是最简单的机器学习分类算法之一，它体现了"数据驱动"方法的基本思想。

**基本思想**：对于新的数据点，找到训练集中距离最近的K个样本，采用多数投票决定类别。

**算法直观理解**：

- **类比**：想象你搬到一个新城市，想知道某个地区的治安情况。一个简单的方法是询问住在附近的K个邻居，如果大多数人说治安良好，你就可以认为这个地区治安不错。
- **在图像分类中**：如果一张新图像与训练集中的5张"猫"图像最相似，那么它很可能也是猫。

![00:02:55](../../../assets/lecture-3/screenshot-01KG66PQV6V9DJBX2KCZJRFZHG.png) [00:02:55](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**距离度量**：

距离度量定义了"相似性"的数学含义。两种最常用的距离度量是：

1. **L1距离（曼哈顿距离）**： $$d_{L1}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|$$

   **几何意义**：

   - 就像在曼哈顿街道上行走，只能沿着街道（水平或垂直）移动，不能斜穿
   - 计算每个维度差异的绝对值之和

   **示例**：

   ```python
   # 两个3D点
   x = [1, 2, 3]
   y = [4, 5, 6]
   
   # L1距离
   d_L1 = |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
   ```

   **特点**：

   - 对异常值不太敏感
   - 在某些坐标系统下更自然（如网格世界）

2. **L2距离（欧氏距离）**： $$d_{L2}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

   **几何意义**：

   - 日常生活中最直观的"直线距离"
   - 两点之间最短路径的长度

   **示例**：

   ```python
   # 使用上面相同的点
   d_L2 = sqrt((1-4)² + (2-5)² + (3-6)²) 
        = sqrt(9 + 9 + 9) 
        = sqrt(27) 
        ≈ 5.20
   ```

   **特点**：

   - 旋转不变性
   - 对大的差异（异常值）更敏感（因为平方）

**L1 vs L2 的可视化理解**：

- **L1距离**：等距离的点形成菱形（在2D中）或超立方体（在高维中）
- **L2距离**：等距离的点形成圆形（在2D中）或超球体（在高维中）

**超参数选择**：

KNN有两个主要超参数需要选择：

1. K值

   ：使用多少个最近邻

   - K=1：使用最近的1个邻居（可能对噪声敏感）
   - K=5：使用最近的5个邻居（更稳定，但可能过度平滑）

2. **距离度量**：L1还是L2

**数据集划分策略**：

为了科学地选择超参数，我们需要合理的数据集划分：

- **训练集（Train Set）**：用于训练模型（在KNN中，就是存储这些数据）
- **验证集（Validation Set）**：用于选择超参数
- **测试集（Test Set）**：用于最终评估模型性能

> **重要原则**：永远不要在测试集上选择超参数！测试集应该只用一次，用于报告最终性能。

**超参数选择流程**：

```python
# 伪代码
best_k = None
best_accuracy = 0

for k in [1, 3, 5, 7, 10]:
    for distance_metric in ['L1', 'L2']:
        # 在训练集上训练（对KNN来说就是存储数据）
        knn = KNN(k=k, metric=distance_metric)
        knn.fit(X_train, y_train)
        
        # 在验证集上评估
        accuracy = knn.score(X_val, y_val)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_metric = distance_metric

# 使用最佳超参数在测试集上评估
final_accuracy = knn.score(X_test, y_test)
```

![00:04:22](../../../assets/lecture-3/screenshot-01KG66S50V77HW9AX1Z199PA2E.png) [00:04:22](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:05:05](../../../assets/lecture-3/screenshot-01KG66TAXWW1G45HEVDKZWWVZ4.png) [00:05:05](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**KNN的优缺点**：

**优点**：

- 概念简单，易于理解
- 训练"过程"非常快（只需存储数据）
- 可以处理多分类问题
- 对数据分布没有假设

**缺点**：

- 预测时间长（需要计算与所有训练样本的距离）
- 在高维空间中表现不佳（维度诅咒）
- 对特征尺度敏感（需要归一化）
- 存储开销大（需要保存所有训练数据）
- 在图像分类中效果一般（像素级距离不能很好地捕捉语义相似性）

### 1.4 线性分类器

线性分类器是深度学习的基础构建块。虽然它本身能力有限，但理解线性分类器是理解神经网络的第一步。

**模型公式**： $$f(\mathbf{x}, \mathbf{W}) = \mathbf{W}\mathbf{x} + \mathbf{b}$$

其中：

- $\mathbf{x}$：输入图像（展平为向量，大小为 $32 \times 32 \times 3 = 3072$）
- $\mathbf{W}$：权重矩阵（大小为 $C \times D$，C为类别数，D为特征维度）
- $\mathbf{b}$：偏置向量（大小为 $C$）

**详细解释**：

**输入向量 $\mathbf{x}$**：

```python
# 原始图像：32×32×3的数组
image = np.array([[[r, g, b], ...], ...])  # shape: (32, 32, 3)

# 展平成向量
x = image.reshape(-1)  # shape: (3072,)
# x = [r₁, g₁, b₁, r₂, g₂, b₂, ..., r₁₀₂₄, g₁₀₂₄, b₁₀₂₄]
```

**权重矩阵 $\mathbf{W}$**：

```python
# 假设有10个类别（如CIFAR-10）
W = np.random.randn(10, 3072)  # shape: (10, 3072)

# W的每一行对应一个类别
# W[0, :] = 飞机的权重模板
# W[1, :] = 汽车的权重模板
# ...
# W[9, :] = 卡车的权重模板
```

**计算得分**：

```python
scores = W @ x + b  # shape: (10,)
# scores = [s_airplane, s_car, s_bird, s_cat, s_deer, 
#           s_dog, s_frog, s_horse, s_ship, s_truck]

# 预测类别：得分最高的类
predicted_class = np.argmax(scores)
```

**三种理解视角**：

理解线性分类器的不同视角有助于建立直觉。

1. **代数视角：矩阵乘法**

   从纯数学角度，这就是一个矩阵-向量乘法：
   $$
   \begin{bmatrix} s_1 \ s_2 \ \vdots \ s_C \end{bmatrix}
   
   \begin{bmatrix} w_{1,1} & w_{1,2} & \cdots & w_{1,D} \ w_{2,1} & w_{2,2} & \cdots & w_{2,D} \ \vdots & \vdots & \ddots & \vdots \ w_{C,1} & w_{C,2} & \cdots & w_{C,D} \end{bmatrix} \begin{bmatrix} x_1 \ x_2 \ \vdots \ x_D \end{bmatrix} + \begin{bmatrix} b_1 \ b_2 \ \vdots \ b_C \end{bmatrix}
   $$
   每个得分 $s_i$ 是第$i$行权重与输入的点积： $$s_i = \mathbf{w}*i \cdot \mathbf{x} + b_i = \sum*{j=1}^{D} w_{i,j} x_j + b_i$$

2. **模板视角：权重作为原型**

   可以将每一行权重重新排列成图像形状并可视化：

   ```python
   # 将第0行权重（飞机类）重新排列成图像
   template_airplane = W[0, :].reshape(32, 32, 3)
   # 可以显示这个图像
   plt.imshow(template_airplane)
   ```

   **直观理解**：

   - 每个类别的权重就像该类别的"模板"或"原型"
   - 分类过程就是计算输入图像与各个模板的相似度
   - 点积 $\mathbf{w}_i \cdot \mathbf{x}$ 衡量了 $\mathbf{x}$ 与模板 $\mathbf{w}_i$ 的相似程度
   - 得分高意味着输入图像与该类模板相似

   **局限性**：

   - 每个类只能学习一个模板
   - 无法捕捉类内的多样性
   - 例如：马可能面向左或面向右，但线性分类器只能学习一个平均的"马"模板

3. **几何视角：决策边界**

   在特征空间中，每个类的权重定义了一个超平面。

   **决策边界**是使得得分为零的点的集合： $$\mathbf{w}_i \cdot \mathbf{x} + b_i = 0$$

   **2D示例**：

   ```python
   # 二分类问题（C=2），2维输入（D=2）
   # 决策边界是一条直线
   # w[0]*x[0] + w[1]*x[1] + b = 0
   # 可以改写为: x[1] = -(w[0]/w[1])*x[0] - b/w[1]
   ```

   **几何特性**：

   - 权重向量 $\mathbf{w}_i$ 垂直于决策边界
   - 权重向量指向该类的正方向
   - 偏置 $b_i$ 控制决策边界与原点的距离

   **可视化**：

   - 在2D平面上，决策边界是直线
   - 在3D空间中，决策边界是平面
   - 在高维空间中，决策边界是超平面

**局限性**：线性分类器只能学习线性决策边界，无法处理非线性可分的数据

**经典的非线性可分问题：XOR问题**：

```python
# XOR数据
# 类别0: (0,0), (1,1)
# 类别1: (0,1), (1,0)

# 无论如何画一条直线，都无法完美分开这两类
# 这就是为什么我们需要神经网络（非线性分类器）
```

**实际例子**：

- **可以处理**：区分"白天"和"夜晚"的图像（基于平均亮度）
- **无法处理**：区分"面向左的马"和"面向右的马"（需要非线性变换）

![00:06:31](../../../assets/lecture-3/screenshot-01KG66WQNAH1SN6ZN1RPJBYGB0.png) [00:06:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

------

## 2. 损失函数

### 2.1 损失函数的定义

有了模型（如线性分类器）后，我们需要一种方法来量化模型的好坏。这就是**损失函数**的作用。

**损失函数（Loss Function）** 量化了模型预测与真实标签之间的差距。损失值越小，说明模型表现越好。

**数学定义**： $$L = \frac{1}{N} \sum_{i=1}^{N} L_i(f(\mathbf{x}_i, \mathbf{W}), y_i) + \lambda R(\mathbf{W})$$

**详细解释各个组成部分**：

1. **数据损失（Data Loss）**：$\frac{1}{N} \sum_{i=1}^{N} L_i$

   - 衡量模型对训练数据的拟合程度
   - $N$ 是训练样本总数
   - $L_i$ 是第$i$个样本的损失
   - $f(\mathbf{x}_i, \mathbf{W})$ 是模型对第$i$个样本的预测
   - $y_i$ 是第$i$个样本的真实标签
   - 取平均是为了让损失值不依赖于数据集大小

   **直观理解**：

   - 如果模型完美预测所有训练样本，数据损失为0
   - 如果模型预测很差，数据损失很大
   - 通过最小化数据损失，我们让模型"记住"训练数据

2. **正则化损失（Regularization Loss）**：$\lambda R(\mathbf{W})$

   - 防止模型过拟合
   - $R(\mathbf{W})$ 是正则化项，通常是权重的某种范数
   - $\lambda$ 是正则化强度（超参数），控制正则化的重要性

   **直观理解**：

   - 限制模型的复杂度
   - 防止权重变得过大
   - 鼓励模型学习简单的模式

   **为什么需要正则化**：

   - 没有正则化，模型可能完美拟合训练数据（包括噪声）
   - 但在新数据上表现糟糕
   - 正则化迫使模型"在训练数据上表现稍差，但在测试数据上表现更好"

其中：

- **数据损失（Data Loss）**：$\frac{1}{N} \sum_{i=1}^{N} L_i$ - 衡量模型对训练数据的拟合程度
- **正则化损失（Regularization Loss）**：$\lambda R(\mathbf{W})$ - 防止过拟合
- $\lambda$：正则化强度（超参数）

**损失函数的作用**：

1. **评估模型**：给定一组权重，损失函数告诉我们这组权重有多好
2. **优化目标**：训练的目标就是找到使损失最小的权重
3. **训练信号**：损失的梯度告诉我们如何调整权重

### 2.2 Softmax损失（交叉熵损失）

Softmax损失（也称交叉熵损失）是多分类问题中最常用的损失函数。

**Softmax函数**：将任意实数向量转换为概率分布

$$P(y_i = k | \mathbf{x}*i) = \frac{e^{s_k}}{\sum*{j} e^{s_j}}$$

**详细解释**：

1. **输入**：模型的原始得分 $\mathbf{s} = [s_1, s_2, \ldots, s_C]$
   - 这些得分可以是任意实数（负数、零、正数）
   - 不同类别的得分可以相差很大
2. **指数化**：$e^{s_k}$
   - 将所有得分转换为正数
   - 放大差异（高得分变得更高，低得分变得更低）
3. **归一化**：除以 $\sum_{j} e^{s_j}$
   - 确保所有概率之和为1
   - 现在每个类别都有一个0到1之间的概率

**数值稳定性技巧**：

```python
# 朴素实现（可能数值不稳定）
def softmax_naive(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)

# 稳定版本（减去最大值）
def softmax_stable(scores):
    scores_shifted = scores - np.max(scores)  # 防止exp溢出
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores)

# 示例
scores = np.array([1000, 1001, 1002])
# 朴素实现会溢出
# 稳定实现工作正常
```

**为什么减去最大值有效**： $$\frac{e^{s_k}}{\sum_j e^{s_j}} = \frac{e^{s_k - M}}{\sum_j e^{s_j - M}}$$ 其中 $M = \max(s_j)$。这个等式在数学上完全等价，但数值上更稳定。

**交叉熵损失**： $$L_i = -\log P(y_i = y_{true} | \mathbf{x}*i) = -\log \left(\frac{e^{s*{y_{true}}}}{\sum_{j} e^{s_j}}\right)$$

**详细推导**： $$L_i = -\log \left(\frac{e^{s_{y_{true}}}}{\sum_{j} e^{s_j}}\right) = -s_{y_{true}} + \log\left(\sum_{j} e^{s_j}\right)$$

**直观解释**：

1. **当预测正确且置信度高时**：

   ```python
   scores = [10, -5, -5]  # 假设真实类别是0
   # Softmax: [0.9999, 0.00005, 0.00005]
   # Loss: -log(0.9999) ≈ 0.0001  # 非常小
   ```

2. **当预测错误时**：

   ```python
   scores = [-5, 10, -5]  # 假设真实类别是0
   # Softmax: [0.00005, 0.9999, 0.00005]
   # Loss: -log(0.00005) ≈ 9.9  # 很大
   ```

3. **当不确定时**：

   ```python
   scores = [0, 0, 0]  # 假设真实类别是0
   # Softmax: [0.333, 0.333, 0.333]
   # Loss: -log(0.333) ≈ 1.1  # 中等
   ```

**特性**：

- 当正确类别概率高时，损失低
  - 概率接近1 → log接近0 → 负log接近0
- 当正确类别概率低时，损失高
  - 概率接近0 → log趋向负无穷 → 负log趋向正无穷
- 鼓励模型输出高置信度的正确预测
  - 不仅要预测对，还要有高置信度
  - 模型被迫"诚实"地表达不确定性

**完整示例**：

```python
def cross_entropy_loss(scores, true_label):
    """
    scores: 模型输出的得分向量, shape (C,)
    true_label: 真实类别的索引, 整数 0 到 C-1
    """
    # 数值稳定的softmax
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores)
    
    # 交叉熵损失
    loss = -np.log(probs[true_label])
    
    return loss, probs

# 示例
scores = np.array([3.0, 1.0, 0.2])
true_label = 0  # 真实类别是第0类

loss, probs = cross_entropy_loss(scores, true_label)
print(f"概率分布: {probs}")  # [0.880, 0.109, 0.011]
print(f"损失: {loss:.4f}")    # 0.1278
```

**与其他损失函数的对比**：

**SVM损失（Hinge Loss）**： $$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

- 关注"边界"：只要正确类别得分比其他类高出足够（$\Delta$），就满足了
- 不关心具体概率值

**Softmax损失的优势**：

- 提供概率解释
- 永远鼓励正确类概率更高（没有"满足点"）
- 更适合需要置信度估计的任务

------

## 3. 正则化（Regularization）

正则化是机器学习中防止过拟合的核心技术之一。理解正则化对于训练好的模型至关重要。

### 3.1 正则化的动机

**核心思想**：在训练数据上表现稍差，但在测试数据上表现更好。

这看起来违反直觉，但实际上是机器学习的核心洞察之一。

**过拟合问题**：

**什么是过拟合**：

- 模型在训练数据上表现完美（甚至100%准确率）
- 但在新数据上表现很差
- 模型"记住"了训练数据，包括噪声和不相关的细节
- 而不是学习到真正的底层规律

**为什么会过拟合**：

1. **模型过于复杂**：参数太多，可以拟合任意复杂的函数
2. **训练数据太少**：没有足够数据约束模型
3. **训练时间过长**：模型开始拟合噪声

**直观例子**：

想象你要学习识别苹果：

**过拟合的学习方式**：

- "这个苹果是红色的，有这个特定的斑点图案，在这个特定的光照下..."
- 记住了每个训练样本的具体细节
- 见到新苹果时（可能是绿色的，或斑点图案不同），无法识别

**好的学习方式（有正则化）**：

- "苹果通常是圆形的，有果柄，表面光滑..."
- 学习到了苹果的一般特征
- 能够泛化到新的苹果

![00:11:54](../../../assets/lecture-3/screenshot-01KG675NF37KTXS94GJ9746GSC.png) [00:11:54](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**多项式拟合示例**：

假设我们有一些带噪声的数据点，想用多项式拟合：

- **低阶多项式（如直线）**：
  - 简单模型
  - 可能欠拟合（训练误差和测试误差都高）
  - 但泛化能力强
- **高阶多项式（如9次多项式）**：
  - 复杂模型
  - 可能过拟合（训练误差低，测试误差高）
  - 通过每个训练点，但在训练点之间剧烈震荡

![00:13:04](../../../assets/lecture-3/screenshot-01KG677M1DR49XQD000PQ2CRXR.png) [00:13:04](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:13:18](../../../assets/lecture-3/screenshot-01KG6780SP9A2Z9NN945RGWP7M.png) [00:13:18](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**直观例子**：

- **模型F1**：完美拟合所有训练点（可能过拟合）
  - 曲线蜿蜒曲折，通过每个训练点
  - 训练误差 = 0
  - 但对新数据的预测不可靠
- **模型F2**：更简单的模型，训练误差稍高，但泛化能力更强
  - 平滑的曲线，捕捉整体趋势
  - 训练误差 > 0
  - 但对新数据的预测更可靠

**奥卡姆剃刀原则（Occam's Razor）**：

> "如果有多个假设都能解释观察到的现象，应该选择最简单的那个。"

在机器学习中的应用：

- 在多个表现相似的模型中，选择最简单的
- 简单模型更可能泛化
- 正则化帮助我们自动实现这一原则

![00:14:19](../../../assets/lecture-3/screenshot-01KG679PRKFRZ9HY5QECY72TR8.png) [00:14:19](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**偏差-方差权衡（Bias-Variance Tradeoff）**：

这是理解正则化的另一个重要视角：

- **偏差（Bias）**：模型的平均预测与真实值之间的差距
  - 高偏差 → 欠拟合
  - 模型太简单，无法捕捉数据的复杂性
- **方差（Variance）**：模型预测的变化程度
  - 高方差 → 过拟合
  - 模型对训练数据的微小变化过于敏感
- **总误差 = 偏差² + 方差 + 不可约误差**

正则化通过增加偏差来减少方差，从而降低总误差。

![00:14:59](../../../assets/lecture-3/screenshot-01KG67ATG66Q150QV9PTGT6X96.png) [00:14:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 3.2 L2正则化（权重衰减）

L2正则化（也称为权重衰减或岭回归）是最常用的正则化方法之一。

**定义**： $$R(\mathbf{W}) = \sum_{i,j} W_{i,j}^2 = ||\mathbf{W}||_2^2$$

**详细解释**：

1. **计算方式**：

   - 对权重矩阵的每个元素平方
   - 将所有平方项相加
   - 这就是权重矩阵的L2范数的平方

2. **完整损失函数**： $$L_{total} = L_{data} + \lambda \sum_{i,j} W_{i,j}^2$$

   其中 $\lambda$ 控制正则化强度

**特性**：

1. **惩罚大的权重值**

   ```python
   # 示例
   w1 = 10.0    # R(w1) = 100
   w2 = 0.1     # R(w2) = 0.01
   # 大权重的惩罚是小权重的10000倍！
   ```

2. **偏好分散的权重分布**

   考虑两组产生相同输出的权重：

   ```python
   # 权重组1：集中在一个大值
   w1 = [10, 0, 0, 0]
   R(w1) = 10² = 100
   
   # 权重组2：分散在多个小值
   w2 = [2.5, 2.5, 2.5, 2.5]
   R(w2) = 4 × 2.5² = 25
   
   # L2正则化偏好w2
   ```

3. **由于平方操作，小权重值的惩罚更小**

   ```python
   w1 = 0.01    # R(w1) = 0.0001
   w2 = 0.001   # R(w2) = 0.000001
   # 从0.01减小到0.001，惩罚减小了100倍
   ```

4. **导致权重值普遍较小但非零**

   - L2正则化很少将权重推向精确的零
   - 而是使权重整体变小
   - 这对某些应用（如特征选择）可能不太理想

**梯度**： $$\frac{\partial R}{\partial W_{i,j}} = 2W_{i,j}$$

**梯度下降更新**（包含L2正则化）： $$W_{i,j} \leftarrow W_{i,j} - \alpha \frac{\partial L_{data}}{\partial W_{i,j}} - 2\alpha\lambda W_{i,j}$$ $$= W_{i,j}(1 - 2\alpha\lambda) - \alpha \frac{\partial L_{data}}{\partial W_{i,j}}$$

**为什么叫"权重衰减"**：

- 看上面的更新规则
- 每次更新时，权重都乘以 $(1 - 2\alpha\lambda)$
- 这个因子小于1，所以权重在每次迭代时都会"衰减"一点

**实现示例**：

```python
# 在训练循环中
for epoch in range(num_epochs):
    # 前向传播
    scores = W @ X + b
    
    # 计算损失（包含L2正则化）
    data_loss = cross_entropy_loss(scores, y)
    reg_loss = 0.5 * lambda_reg * np.sum(W * W)  # 0.5是为了梯度方便
    total_loss = data_loss + reg_loss
    
    # 反向传播
    dW_data = compute_data_gradient(...)  # 数据损失的梯度
    dW_reg = lambda_reg * W               # 正则化的梯度
    dW = dW_data + dW_reg                 # 总梯度
    
    # 更新权重
    W -= learning_rate * dW
```

**什么时候使用L2正则化**：

- 默认选择，几乎总是有帮助
- 当你希望所有特征都对预测有贡献时
- 当你的权重不需要稀疏时

### 3.3 L1正则化

L1正则化（也称为Lasso正则化）是另一种流行的正则化方法。

**定义**： $$R(\mathbf{W}) = \sum_{i,j} |W_{i,j}| = ||\mathbf{W}||_1$$

**详细解释**：

1. 计算方式

   ：

   - 对权重矩阵的每个元素取绝对值
   - 将所有绝对值相加
   - 这就是权重矩阵的L1范数

**特性**：

1. **偏好稀疏的权重分布**

   "稀疏"意味着许多权重精确等于零。

   ```python
   # L1正则化倾向于产生这样的权重：
   w_sparse = [0, 0, 5.0, 0, 3.0, 0, 0, 2.0]
   # 许多权重恰好为0
   
   # 而L2正则化倾向于产生：
   w_dense = [0.1, 0.3, 4.5, 0.2, 2.8, 0.1, 0.2, 1.9]
   # 权重都很小，但很少精确为0
   ```

2. **导致许多权重值精确为0**

   **为什么会这样**：

   - L1的梯度是常数（+1或-1）
   - 不像L2，梯度不随权重大小变化
   - 小权重和大权重受到相同的推向零的力
   - 所以权重更容易被推到精确的零

3. **可用于特征选择**

   **特征选择**：确定哪些输入特征对预测重要。

   ```python
   # 训练后的权重
   W = [0, 0, 5.0, 0, 3.0, 0, 0, 2.0]
   
   # 解释：
   # 特征0,1,3,5,6不重要（权重为0）
   # 特征2,4,7重要（权重非零）
   
   # 这在高维数据中特别有用
   # 可以自动发现重要特征
   ```

**梯度**： $$\frac{\partial R}{\partial W_{i,j}} = \text{sign}(W_{i,j}) = \begin{cases} +1 & \text{if } W_{i,j} > 0 \ -1 & \text{if } W_{i,j} < 0 \ \text{undefined} & \text{if } W_{i,j} = 0 \end{cases}$$

**实践中的处理**：

- 在$W_{i,j} = 0$处，梯度未定义
- 实践中通常取0或使用次梯度（subgradient）

**实现示例**：

```python
def l1_regularization_gradient(W):
    """
    计算L1正则化的梯度
    """
    return np.sign(W)

# 在训练循环中
for epoch in range(num_epochs):
    # ... 前向传播 ...
    
    # L1正则化损失
    reg_loss = lambda_reg * np.sum(np.abs(W))
    
    # 梯度
    dW_data = compute_data_gradient(...)
    dW_reg = lambda_reg * np.sign(W)  # L1的梯度
    dW = dW_data + dW_reg
    
    # 更新
    W -= learning_rate * dW
```

![00:15:29](../../../assets/lecture-3/screenshot-01KG67BN53BS2QQV576H3ZN8M8.png) [00:15:29](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**L1 vs L2 的几何解释**：

在2D空间中可视化：

**L2正则化**：

- 约束区域是圆形
- 等高线也是圆形
- 最优点通常不在坐标轴上
- 权重很少精确为零

**L1正则化**：

- 约束区域是菱形（有尖角）
- 最优点容易落在坐标轴上（尖角处）
- 导致某些权重精确为零

![00:18:21](../../../assets/lecture-3/screenshot-01KG67GDZCW7FGZZRZR9332PPQ.png) [00:18:21](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:18:31](../../../assets/lecture-3/screenshot-01KG67GPKKWCNQTSZ6HWD8KQ2C.png) [00:18:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**什么时候使用L1正则化**：

- 当你需要稀疏解时（特征选择）
- 当你怀疑许多特征不重要时
- 当模型可解释性很重要时

### 3.4 L1 vs L2 对比示例

让我们通过具体例子来理解L1和L2的区别。

考虑两组权重，它们产生相同的预测得分：

- $\mathbf{w}_1 = [1, 0, 0, 0]$
- $\mathbf{w}_2 = [0.25, 0.25, 0.25, 0.25]$

**假设输入** $\mathbf{x} = [1, 1, 1, 1]$：

```python
# 两组权重产生相同的输出
output1 = np.dot([1, 0, 0, 0], [1, 1, 1, 1]) = 1
output2 = np.dot([0.25, 0.25, 0.25, 0.25], [1, 1, 1, 1]) = 1
```

**L2正则化**：

计算L2正则化项：

- $R(\mathbf{w}_1) = 1^2 + 0^2 + 0^2 + 0^2 = 1$
- $R(\mathbf{w}_2) = 4 \times (0.25)^2 = 4 \times 0.0625 = 0.25$

**结论**：**L2偏好** $\mathbf{w}_2$（更分散的权重）

**为什么**：

- $\mathbf{w}_2$ 的正则化损失是 $\mathbf{w}_1$ 的1/4
- L2惩罚集中的权重
- 鼓励将"重要性"分散到多个特征

**L1正则化**：

计算L1正则化项：

- $R(\mathbf{w}_1) = |1| + |0| + |0| + |0| = 1$
- $R(\mathbf{w}_2) = 4 \times |0.25| = 1$

**结论**：在正则化损失方面，两者相同（都是1）

**但在实践中**：

- L1倾向于产生更稀疏的解
- 所以可能更偏好 $\mathbf{w}_1$
- 因为它只用了一个特征

![00:20:20](../../../assets/lecture-3/screenshot-01KG67KQ66ZM7JVNXG7A22Y6RN.png) [00:20:20](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**更极端的例子**：

如果一组权重是 $\mathbf{w}_3 = [0.9, 0.1, 0, 0]$（假设仍产生相同输出）：

**L2正则化**： $$R(\mathbf{w}_3) = 0.9^2 + 0.1^2 = 0.81 + 0.01 = 0.82$$

**L1正则化**： $$R(\mathbf{w}_3) = 0.9 + 0.1 = 1.0$$

L1会倾向于将0.1这个小权重推向0，产生 $[0.9, 0, 0, 0]$（更稀疏）。

![00:20:37](../../../assets/lecture-3/screenshot-01KG67M6CTXCM5M5XEH541D7RW.png) [00:20:37](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Elastic Net（结合L1和L2）**：

有时我们同时使用L1和L2： $$R(\mathbf{W}) = \lambda_1 ||\mathbf{W}||_1 + \lambda_2 ||\mathbf{W}||_2^2$$

**优点**：

- 结合了两者的优势
- L1提供稀疏性
- L2提供稳定性和分散性
- 在高维数据中特别有用

![00:21:43](../../../assets/lecture-3/screenshot-01KG67P6NH5SF85EREJ35E3PJE.png) [00:21:43](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:26:32](../../../assets/lecture-3/screenshot-01KG67Z1NHSN1QBJT7X41KX68R.png) [00:26:32](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 3.5 正则化的作用

正则化在机器学习中扮演多重角色：

1. **表达模型偏好**

   通过选择不同的正则化项，我们可以表达对解的偏好：

   - **L2**："我希望所有特征都有贡献，没有某个特征占主导"
   - **L1**："我希望只有少数关键特征起作用"
   - **其他正则化**：可以编码各种先验知识

   **示例**：图像去噪

   ```python
   # 全变分（Total Variation）正则化
   # 鼓励相邻像素相似（平滑图像）
   R(image) = sum(|image[i,j] - image[i+1,j]| + 
                   |image[i,j] - image[i,j+1]|)
   ```

2. **提高泛化能力**

   **机制**：

   - 限制模型复杂度
   - 防止过拟合训练数据
   - 迫使模型学习简单、泛化的模式

   **效果**：

   - 训练误差可能稍高
   - 但测试误差显著降低
   - 整体性能提升

   **实验证据**：

   ```python
   # 没有正则化
   # 训练误差: 0.05, 测试误差: 0.30
   
   # 有L2正则化 (λ=0.01)
   # 训练误差: 0.10, 测试误差: 0.15
   
   # 我们接受更高的训练误差以换取更好的测试性能
   ```

3. **改善优化**

   **L2正则化的额外好处**：

   - 使损失函数更凸（convex）
   - 改善条件数（condition number）
   - 加速收敛
   - 使梯度下降更稳定

   **数学原因**：

   - L2正则化添加的 $||\mathbf{W}||^2$ 项是强凸的
   - 结合后的损失函数更"碗状"
   - 梯度下降更容易找到最小值

   **实践影响**：

   ```python
   # 没有正则化：可能需要10000次迭代收敛
   # 有L2正则化：可能只需要5000次迭代
   ```

**其他类型的正则化**：

本课程后面会介绍更多正则化技术：

1. **Dropout**：随机丢弃神经元
2. **Batch Normalization**：标准化激活值
3. **Data Augmentation**：增加训练数据的多样性
4. **Early Stopping**：在验证误差开始上升时停止训练

**选择正则化强度 $\lambda$**：

这是一个需要调节的超参数：

```python
# 使用验证集选择最佳 lambda
lambda_candidates = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
best_lambda = None
best_val_acc = 0

for lambda_reg in lambda_candidates:
    model = train_model(X_train, y_train, lambda_reg)
    val_acc = evaluate(model, X_val, y_val)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_lambda = lambda_reg

print(f"最佳 λ: {best_lambda}")
```

**常见误区**：

1. **误区**："正则化总是有帮助的"
   - **事实**：如果 $\lambda$ 太大，会导致欠拟合
   - 需要调节以找到最佳值
2. **误区**："正则化会降低训练准确率"
   - **事实**：正确，但这是预期的
   - 目标是提高测试准确率，不是训练准确率
3. **误区**："对所有参数使用相同的正则化"
   - **事实**：通常不对偏置项进行正则化
   - 只正则化权重 $\mathbf{W}$，不正则化偏置 $\mathbf{b}$

```python
# 正确的做法
reg_loss = lambda_reg * (np.sum(W**2))  # 只对W正则化
# 不包括 b
```

------

## 4. 优化（Optimization）

有了损失函数和正则化后，下一个问题是：如何找到最好的参数？这就是优化要解决的问题。

### 4.1 优化问题

**目标**：找到使损失函数最小的权重 $\mathbf{W}^*$

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} L(\mathbf{W})$$

**详细解释**：

- $\arg\min$：返回使函数达到最小值的参数
- $L(\mathbf{W})$：包含数据损失和正则化损失的总损失
- $\mathbf{W}^*$：最优权重（我们要找的）

**为什么这是困难的**：

1. **高维空间**

   - 现代深度学习模型可能有数百万甚至数十亿参数
   - 在如此高维空间中搜索最小值极其困难

   **示例规模**：

   ```python
   # 一个中等大小的ResNet
   num_parameters = 25,000,000
   # 这是一个25,000,000维的优化问题！
   ```

2. **非凸损失函数**

   - 神经网络的损失函数通常是非凸的
   - 存在许多局部最小值
   - 没有保证能找到全局最优解

3. **计算成本**

   - 评估损失函数可能很昂贵
   - 特别是当数据集很大时
   - 需要高效的优化算法

### 4.2 损失景观（Loss Landscape）

为了理解优化，我们使用"损失景观"这个强大的可视化比喻。

**可视化比喻**：

想象一个真实的地理景观：

- **垂直轴（Z轴）**：海拔高度 → 在我们的问题中是**损失值**
- **水平轴（X, Y轴）**：地理位置 → 在我们的问题中是**模型参数**
- **目标**：找到景观中的最低点（山谷底部）→ 找到**最小损失**

![00:26:54](../../../assets/lecture-3/screenshot-01KG67ZPE5KHBCTZE27090Z22G.png) [00:26:54](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**关键特征**：

1. **山峰和山谷**
   - 山峰：损失高的区域（糟糕的参数）
   - 山谷：损失低的区域（好的参数）
   - 最深的山谷：全局最小值（最好的参数）
2. **平坦区域和陡峭区域**
   - 平坦：损失变化缓慢（梯度小）
   - 陡峭：损失变化剧烈（梯度大）
3. **局部最小值**
   - 周围比它高，但不是全局最低
   - 优化可能被"困住"
4. **鞍点**
   - 某些方向是最小值，其他方向是最大值
   - 就像马鞍的形状
   - 在高维空间中非常常见

**关键限制**：

**我们是"蒙眼"的**：

- 无法"看到"整个景观
- 只能感知当前位置的局部信息
- 就像在浓雾中登山，只能感觉脚下的坡度
- 必须依靠局部梯度信息

**实际含义**：

- 不能直接跳到最低点
- 必须一步一步地移动
- 每一步只能依据当前位置的斜率

**高维的挑战**：

在实际深度学习中：

- 景观不是2D或3D，而是数百万维
- 无法真正可视化
- 但"沿着斜率下降"的直觉仍然适用

### 4.3 策略1：随机搜索（不推荐）

让我们从最简单（但最糟糕）的方法开始：随机猜测。

**算法**：

```python
# 伪代码
best_W = None
best_loss = float('inf')

for i in range(1000):
    W_random = np.random.randn(D, C)
    loss = compute_loss(W_random, X_train, y_train)
    if loss < best_loss:
        best_W = W_random
        best_loss = loss
```

**详细解释**：

1. 随机生成一组权重
2. 计算这组权重的损失
3. 如果比目前最好的还好，就保存它
4. 重复1000次
5. 返回找到的最好权重

**性能**：CIFAR-10上约15.5%准确率（远低于99.7%的最优水平）

**为什么这么糟糕**：

1. **搜索空间太大**

   ```python
   # 假设每个参数有100个可能的"好"值
   # 10个参数就有100^10 = 10^20种组合
   # 1000次尝试根本不够
   ```

2. **没有利用梯度信息**

   - 完全忽略了损失函数的结构
   - 不知道哪个方向更有希望

3. **效率极低**

   - 即使运行很长时间，也很难找到好解

**什么时候可能有用**：

- 仅作为基线（baseline）来比较其他方法
- 非常简单的问题（参数很少）
- 梯度无法计算的情况（罕见）

**比喻**：

- 就像蒙着眼睛在山区随机传送
- 偶尔可能传送到低谷
- 但大多数时候在山峰上
- 完全没有利用地形信息

### 4.4 策略2：跟随斜率（梯度下降）

这是机器学习中的核心优化策略。

**核心思想**：

- 感知当前位置的斜率
- 朝着下坡方向移动
- 重复直到到达谷底

**登山比喻**（更准确的版本）：

1. 蒙着眼睛站在山上
2. 用脚感受地面的倾斜
3. 朝着最陡的下坡方向走一小步
4. 重复2-3，直到周围都是上坡（到达谷底）

**数学形式化**：

**梯度**告诉我们上升最快的方向 **负梯度**告诉我们下降最快的方向

$$\mathbf{W}*{新} = \mathbf{W}*{当前} - \alpha \cdot \nabla_{\mathbf{W}} L(\mathbf{W}_{当前})$$

其中：

- $\alpha$：学习率（步长）
- $\nabla_{\mathbf{W}} L$：损失对权重的梯度

**为什么这样有效**：

1. **利用了函数的局部结构**
   - 梯度是局部最优的移动方向
   - 在光滑函数上保证损失下降
2. **数学保证**（对于小学习率）：
   - 每一步都会减小损失（在凸函数上）
   - 最终会收敛到（局部）最小值
3. **计算高效**：
   - 只需要计算一次梯度
   - 比尝试所有可能方向高效得多

**直观例子**：

想象损失景观是一个碗：

```
     ___________
    /           \
   /             \
  /               \
 /                 \
/___________________\
     ← W当前在这里

梯度指向碗边（上升方向）
负梯度指向碗底（下降方向）
```

每次迭代：

```
迭代1: W = W - α·梯度1 → 向碗底移动一小步
迭代2: W = W - α·梯度2 → 继续向碗底移动
...
迭代N: 到达碗底（梯度≈0）
```

------

## 5. 梯度下降及其变体

现在我们深入探讨梯度下降的数学细节和实际实现。

### 5.1 梯度的数学定义

**一维导数**：

对于单变量函数 $f(x)$： $$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**几何意义**：

- 函数在某点的切线斜率
- 描述函数值如何随输入变化

**示例**：

```python
def f(x):
    return x**2

# 在x=3处的导数
x = 3
h = 0.0001
derivative = (f(x + h) - f(x)) / h
# ≈ 6 (精确值是2*3=6)
```

**多维梯度**：

对于多变量函数 $L(\mathbf{W})$，其中 $\mathbf{W}$ 是矩阵： $$\nabla_{\mathbf{W}} L = \left[\frac{\partial L}{\partial W_{1,1}}, \frac{\partial L}{\partial W_{1,2}}, \ldots, \frac{\partial L}{\partial W_{m,n}}\right]$$

**详细解释**：

- $\nabla$ (nabla)：梯度符号
- 梯度是一个与 $\mathbf{W}$ 同形状的矩阵
- 每个元素 $\frac{\partial L}{\partial W_{i,j}}$ 表示：
  - 如果增加 $W_{i,j}$，损失 $L$ 会如何变化
  - 是 $L$ 对 $W_{i,j}$ 的偏导数

**示例**：

```python
# 假设 W 是 2×3 的矩阵
W = [[w11, w12, w13],
     [w21, w22, w23]]

# 梯度也是 2×3 的矩阵
dL/dW = [[∂L/∂w11, ∂L/∂w12, ∂L/∂w13],
         [∂L/∂w21, ∂L/∂w22, ∂L/∂w23]]

# 每个元素告诉我们：
# "如果我增加w_ij一点点，损失L会增加多少？"
```

**重要性质**：

1. **梯度方向是函数上升最快的方向**

   数学证明（简化版）：

   - 沿任意单位方向 $\mathbf{v}$ 的方向导数：$\nabla L \cdot \mathbf{v}$
   - 由柯西-施瓦茨不等式：$\nabla L \cdot \mathbf{v} \leq ||\nabla L|| \cdot ||\mathbf{v}||$
   - 当 $\mathbf{v}$ 与 $\nabla L$ 同向时，等号成立
   - 所以梯度方向是上升最快的方向

2. **负梯度方向是函数下降最快的方向**

   - 由上述性质直接推导
   - $-\nabla L$ 是下降最快的方向
   - 这就是为什么我们更新 $W - \alpha \nabla L$

**几何直觉**：

在2D等高线图上：

```
    Level sets (等高线)
         ___
        /   \  ← 高损失
       |  *  | ← W当前位置
        \___/  ← 低损失
           
    梯度 ↑ (垂直于等高线，指向外)
    负梯度 ↓ (指向内，朝向中心/最小值)
```

### 5.2 梯度计算方法

有两种主要方法计算梯度：

**方法1：数值梯度（慢，近似）**

基于导数的定义直接计算：

```python
def numerical_gradient(f, W, h=1e-5):
    """
    f: 损失函数
    W: 权重矩阵
    h: 小增量
    """
    grad = np.zeros_like(W)
    
    # 迭代W的每个元素
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = W[ix]
        
        # f(x+h)
        W[ix] = old_value + h
        fxh_pos = f(W)
        
        # f(x-h)
        W[ix] = old_value - h
        fxh_neg = f(W)
        
        # 中心差分
        grad[ix] = (fxh_pos - fxh_neg) / (2*h)
        
        # 恢复原值
        W[ix] = old_value
        it.iternext()
    
    return grad
```

**详细解释**：

1. **中心差分法**： $$\frac{\partial f}{\partial w} \approx \frac{f(w+h) - f(w-h)}{2h}$$
   - 比单边差分 $\frac{f(w+h) - f(w)}{h}$ 更准确
   - 误差是 $O(h^2)$ 而不是 $O(h)$
2. **选择h**：
   - 太大：近似不准确
   - 太小：浮点精度问题
   - 通常用 $h = 10^{-5}$

**优点**：

- 实现简单
- 易于理解
- 适合验证解析梯度的正确性（梯度检查）

**缺点**：

- 非常慢（需要对每个参数调用两次前向传播）
- 只是近似（不是精确值）
- 对于大模型不可行

**示例**：

```python
# 简单的二次函数
def f(w):
    return w**2

w = 3.0
h = 1e-5

# 数值梯度
numerical_grad = (f(w + h) - f(w - h)) / (2*h)
print(f"数值梯度: {numerical_grad}")  # ≈ 6

# 解析梯度
analytical_grad = 2 * w
print(f"解析梯度: {analytical_grad}")  # 精确的6
```

![00:27:33](../../../assets/lecture-3/screenshot-01KG680SA9YBHPMA9X9KTT55H9.png) [00:27:33](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**方法2：解析梯度（快，精确）**

使用微积分和链式法则推导出精确的梯度公式。

**示例：线性分类器的梯度**

对于Softmax损失： $$L = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right) + \lambda \sum_{i,j} W_{i,j}^2$$

**推导步骤**（简化）：

1. **Softmax梯度**： $$\frac{\partial L_i}{\partial s_k} = \begin{cases} p_k - 1 & \text{if } k = y_i \ p_k & \text{otherwise} \end{cases}$$

   其中 $p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}$

2. **链式法则**： $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial s} \cdot \frac{\partial s}{\partial W}$$

   由于 $s = Wx$： $$\frac{\partial s}{\partial W} = x^T$$

3. **最终梯度**： $$\frac{\partial L}{\partial W} = (\mathbf{p} - \mathbf{e}_{y_i}) \otimes \mathbf{x}^T + 2\lambda W$$

   其中 $\mathbf{e}_{y_i}$ 是one-hot向量

**实现**：

```python
def softmax_loss_gradient(W, X, y, lambda_reg):
    """
    计算Softmax损失的梯度
    W: 权重矩阵 (C, D)
    X: 输入数据 (D,) 或 (N, D)
    y: 标签 (整数) 或 (N,)
    """
    # 前向传播
    scores = W @ X  # (C,)
    
    # Softmax
    scores -= np.max(scores)  # 数值稳定
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    
    # 梯度
    dscores = probs.copy()
    dscores[y] -= 1  # 正确类别减1
    
    # 链式法则
    dW = np.outer(dscores, X)  # (C, D)
    
    # 加上正则化梯度
    dW += 2 * lambda_reg * W
    
    return dW
```

**优点**：

- 精确（不是近似）
- 快速（只需一次前向和一次反向传播）
- 可扩展到大型模型

**缺点**：

- 需要推导（可能复杂）
- 容易出错（需要仔细验证）

![00:29:35](../../../assets/lecture-3/screenshot-01KG6845SW06B0WBAH5TRKMFH2.png) [00:29:35](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**梯度检查（Gradient Checking）**：

**最佳实践**：用数值梯度验证解析梯度

```python
def gradient_check(f, W):
    """
    验证解析梯度的正确性
    """
    # 计算解析梯度
    analytical_grad = compute_analytical_gradient(f, W)
    
    # 计算数值梯度
    numerical_grad = numerical_gradient(f, W)
    
    # 比较
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    sum_norm = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    relative_error = diff / (sum_norm + 1e-8)
    
    print(f"相对误差: {relative_error}")
    
    if relative_error < 1e-7:
        print("✓ 梯度正确!")
    elif relative_error < 1e-5:
        print("⚠ 梯度可能正确，但要小心")
    else:
        print("✗ 梯度可能有误!")
    
    return relative_error

# 使用
gradient_check(lambda w: loss_function(w, data), W_init)
```

**相对误差的解释**：

- < $10^{-7}$：完美
- < $10^{-5}$：可以接受
- < $10^{-3}$：可能有问题
- \> $10^{-3}$：肯定有错误

![00:30:11](../../../assets/lecture-3/screenshot-01KG6855GY1MW2J0VPMG4GHK5V.png) [00:30:11](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**推荐使用，并用数值梯度验证**

**工作流程**：

1. 实现解析梯度
2. 实现数值梯度
3. 对比两者（梯度检查）
4. 一旦验证正确，就只使用解析梯度训练

![00:30:50](../../../assets/lecture-3/screenshot-01KG68R314QGSSQK2K6J1P2Y0P.png) [00:30:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:31:35](../../../assets/lecture-3/screenshot-01KG68SBKB1SCA92ZDK28JSC0S.png) [00:31:35](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.3 梯度下降算法

现在我们把所有部分组合起来，实现完整的梯度下降。

**基本算法**：

```python
# 伪代码
while True:
    dW = compute_gradient(W, X, y)  # 计算梯度
    W = W - learning_rate * dW       # 参数更新
```

**详细实现**：

```python
def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000, lambda_reg=0.01):
    """
    完整的梯度下降训练
    
    参数:
    - X: 训练数据 (N, D)
    - y: 标签 (N,)
    - learning_rate: 学习率
    - num_iterations: 迭代次数
    - lambda_reg: 正则化强度
    
    返回:
    - W: 训练好的权重
    - loss_history: 损失历史
    """
    N, D = X.shape
    C = np.max(y) + 1  # 类别数
    
    # 初始化权重（小随机值）
    W = 0.01 * np.random.randn(C, D)
    
    loss_history = []
    
    for it in range(num_iterations):
        # 前向传播：计算损失
        scores = W @ X.T  # (C, N)
        
        # Softmax和交叉熵损失
        scores -= np.max(scores, axis=0)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=0)
        
        # 数据损失
        correct_log_probs = -np.log(probs[y, range(N)])
        data_loss = np.mean(correct_log_probs)
        
        # 正则化损失
        reg_loss = 0.5 * lambda_reg * np.sum(W * W)
        
        # 总损失
        loss = data_loss + reg_loss
        loss_history.append(loss)
        
        # 反向传播：计算梯度
        dscores = probs.copy()
        dscores[y, range(N)] -= 1
        dscores /= N
        
        # 权重梯度
        dW = dscores @ X  # (C, D)
        dW += lambda_reg * W  # 正则化梯度
        
        # 参数更新
        W -= learning_rate * dW
        
        # 打印进度
        if it % 100 == 0:
            print(f"迭代 {it}: 损失 = {loss:.4f}")
    
    return W, loss_history
```

**数学形式**： $$\mathbf{W}_{t+1} = \mathbf{W}*t - \alpha \nabla*{\mathbf{W}} L(\mathbf{W}_t)$$

其中 $\alpha$ 是**学习率（步长）**

**学习率的作用**：

学习率是最重要的超参数之一。

**太小**：

```python
# learning_rate = 0.000001
# 每次只移动一点点
迭代1: loss = 2.5
迭代100: loss = 2.49
迭代1000: loss = 2.4
# 收敛太慢！
```

**太大**：

```python
# learning_rate = 100
# 每次移动太多
迭代1: loss = 2.5
迭代2: loss = 5.0
迭代3: loss = 10.0
# 发散了！
```

**刚好**：

```python
# learning_rate = 0.01
迭代1: loss = 2.5
迭代10: loss = 1.8
迭代50: loss = 0.5
迭代100: loss = 0.3
# 稳定快速收敛
```

**可视化损失曲线**：

```python
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练过程中的损失')
plt.show()

# 好的学习率：平滑下降曲线
# 学习率太大：震荡或上升
# 学习率太小：下降太慢
```

![00:34:01](../../../assets/lecture-3/screenshot-01KG68Y89T430EGN51KPAKGRCK.png) [00:34:01](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**停止条件**：

```python
# 方法1：固定迭代次数
for it in range(1000):
    ...

# 方法2：损失变化很小
tolerance = 1e-6
prev_loss = float('inf')
while True:
    loss = ...
    if abs(prev_loss - loss) < tolerance:
        break
    prev_loss = loss

# 方法3：验证集性能不再提升
best_val_acc = 0
patience = 10
no_improve_count = 0

while no_improve_count < patience:
    # 训练一个epoch
    val_acc = evaluate_on_validation_set()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_count = 0
    else:
        no_improve_count += 1
```

![00:34:55](../../../assets/lecture-3/screenshot-01KG68ZRQC1FAJ970JT9GFPPH6.png) [00:34:55](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:35:34](../../../assets/lecture-3/screenshot-01KG690WN7WAXKJ19SQN1QATBR.png) [00:35:34](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.4 随机梯度下降（SGD）

批量梯度下降（使用全部数据）的问题是：对于大数据集来说太慢了。

**动机**：计算全部数据的梯度开销太大

**问题示例**：

```python
# ImageNet数据集
N = 1,280,000  # 样本数

# 批量梯度下降
# 每次迭代需要：
# - 前向传播: 1,280,000次
# - 反向传播: 1,280,000次
# - 太慢了！
```

**核心观察**：

- 整个数据集的梯度可以看作是每个样本梯度的期望
- 我们可以用一个小批量来近似这个期望
- 虽然有噪声，但期望是正确的

**Mini-batch SGD**：

最常用的梯度下降变体。

```python
# 伪代码
for epoch in range(num_epochs):
    # 随机打乱数据
    shuffle(X_train, y_train)
    
    for i in range(0, N, batch_size):
        # 提取mini-batch
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # 计算梯度（仅在mini-batch上）
        dW = compute_gradient(W, X_batch, y_batch)
        
        # 更新参数
        W = W - learning_rate * dW
```

**详细实现**：

```python
def mini_batch_sgd(X, y, learning_rate=0.01, batch_size=256, 
                    num_epochs=10, lambda_reg=0.01):
    """
    Mini-batch随机梯度下降
    
    参数:
    - batch_size: 每个批次的样本数
    - num_epochs: 遍历整个数据集的次数
    """
    N, D = X.shape
    C = np.max(y) + 1
    
    # 初始化
    W = 0.01 * np.random.randn(C, D)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        # 随机打乱数据
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Mini-batch训练
        num_batches = N // batch_size
        
        for batch_idx in range(num_batches):
            # 获取当前batch
            start = batch_idx * batch_size
            end = start + batch_size
            
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            
            # 前向传播（只在batch上）
            scores = W @ X_batch.T
            
            # ... 计算损失和梯度（与之前相同）...
            
            # 更新参数
            W -= learning_rate * dW
        
        # 计算整个训练集的损失（用于监控）
        full_loss = compute_loss(W, X, y, lambda_reg)
        loss_history.append(full_loss)
        
        print(f"Epoch {epoch}: 损失 = {full_loss:.4f}")
    
    return W, loss_history
```

**关键概念**：

1. **Batch Size**：每次更新使用的样本数

   常见选择：

   - 32
   - 64
   - 128
   - 256
   - 512

   **权衡**：

   ```python
   # 小batch size (如32):
   # 优点：更新频繁，可能更快收敛，内存占用小
   # 缺点：梯度噪声大，可能不稳定
   
   # 大batch size (如512):
   # 优点：梯度更准确，更稳定，GPU利用率高
   # 缺点：内存占用大，可能陷入sharp minima
   ```

2. **Epoch**：遍历整个训练集一次

   ```python
   # 一个epoch的迭代次数
   iterations_per_epoch = N // batch_size
   
   # 示例：N=50000, batch_size=256
   iterations_per_epoch = 50000 // 256 = 195
   ```

3. **Iteration/Step**：一次参数更新

   ```python
   total_iterations = num_epochs * iterations_per_epoch
   
   # 10 epochs, 195 iterations/epoch
   total_iterations = 10 * 195 = 1950
   ```

**优势**：

1. **计算高效**

   ```python
   # 批量GD: 每次迭代处理N=50000个样本
   # Mini-batch SGD: 每次迭代处理256个样本
   # 速度提升约: 50000/256 ≈ 195倍！
   ```

2. **引入随机性有助于逃离局部最优**

   **没有噪声（批量GD）**：

   ```
   陷入局部最小值
        ↓
   ___/‾‾‾\___
      停在这
   ```

   **有噪声（SGD）**：

   ```
   可能跳出局部最小值
        ↓  ↗震荡
   ___/‾‾‾\___
      可能逃离
   ```

3. **在线学习**

   - 可以在新数据到达时更新模型
   - 不需要存储整个数据集

**Mini-batch的噪声**：

这是一个关键特性，既是优点也是缺点：

**噪声的来源**：

```python
# 真实梯度（使用全部数据）
true_gradient = compute_gradient(W, X_all, y_all)

# Mini-batch梯度
batch_gradient = compute_gradient(W, X_batch, y_batch)

# 它们不完全相同！
# batch_gradient = true_gradient + noise
```

**影响**：

- 参数更新不会沿着精确的最优路径
- 路径会"之字形"前进
- 但平均方向是正确的

**可视化**：

```
批量GD:  Start ──→──→──→ Minimum
                直线路径

Mini-batch SGD:  
         Start ↗↘↗↘↗ Minimum
                之字形路径，但最终到达
```

![00:36:58](../../../assets/lecture-3/screenshot-01KG69713PWV5GB4CY757R4NGJ.png) [00:36:58](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Batch Size的影响**：

```python
# 实验：不同batch size的效果
batch_sizes = [32, 64, 128, 256, 512]

for bs in batch_sizes:
    W, history = mini_batch_sgd(X, y, batch_size=bs)
    plt.plot(history, label=f'BS={bs}')

plt.legend()
plt.show()

# 观察：
# - 小BS：噪声大，曲线抖动多，但可能逃离bad local minima
# - 大BS：曲线平滑，但可能陷入sharp minima
```

![00:37:58](../../../assets/lecture-3/screenshot-01KG698NM3RMF42WBNCCFFQT3W.png) [00:37:58](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

------

### 5.5 SGD的问题

虽然SGD是基础算法，但它有一些严重的问题。这些问题激发了更先进的优化器的发展。

**问题1：狭窄山谷中的震荡**

想象一个狭长的山谷：一个方向很陡，另一个方向很平缓。

```
    陡峭方向
    ↑
    |
----+----→ 平缓方向
    |
    ↓
```

**发生的事情**：

```
路径应该是这样的:
    →→→→→→  (沿着山谷前进)

但实际是这样的:
    ↗↘↗↘↗  (上下震荡)
```

**数学解释**：

- 梯度在陡峭方向上很大
- 梯度在平缓方向上很小
- SGD对所有方向使用相同的学习率
- 结果：在陡峭方向上步子太大（震荡），在平缓方向上步子太小（进展慢）
- **在陡峭方向上大幅震荡**
- **在平缓方向上进展缓慢**

**可视化示例**：

```python
# 定义一个狭窄山谷型损失函数
def loss(x, y):
    return 0.5 * x**2 + 50 * y**2  # y方向远比x方向陡峭

# SGD会在y方向震荡，在x方向缓慢前进
```

**条件数（Condition Number）**：

这个问题在数学上与"条件数"有关。

**定义**： $$\kappa = \frac{\lambda_{max}}{\lambda_{min}}$$

其中 $\lambda_{max}$ 和 $\lambda_{min}$ 是Hessian矩阵的最大和最小特征值。

**解释**：

- 高条件数 → 某些方向的曲率远大于其他方向
- 导致SGD在不同方向上的表现差异巨大

**示例**：

```python
# 良好条件（κ≈1）
# 损失函数像一个圆形碗
f(x,y) = x² + y²
# SGD表现良好

# 糟糕条件（κ很大）
# 损失函数像一个狭长山谷
f(x,y) = 0.01·x² + 100·y²
# κ = 100/0.01 = 10000
# SGD会严重震荡
```

- **原因**：条件数（condition number）高 - Hessian矩阵最大/最小特征值比值大

**实际影响**：

- 训练速度慢
- 可能需要仔细调节学习率
- 对于深度神经网络尤其明显

![00:39:13](../../../assets/lecture-3/screenshot-01KG69ARKRKEGYDVBKZ6D7E6R4.png) [00:39:13](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**问题2：局部最优和鞍点**

**局部最优（Local Minimum）**：

```
     全局最优          局部最优
        ↓                ↓
    ___/ \____________/‾\___
                    SGD可能停在这里
```

- **定义**：周围所有点的损失都比它高，但不是全局最低点
- **问题**：SGD到达后，梯度为零，无法继续前进
- **在凸函数中**：不是问题（局部最优=全局最优）
- **在非凸函数中**（如神经网络）：严重问题
- **局部最优**：梯度为零但不是全局最优

**鞍点（Saddle Point）**：

更微妙也更常见的问题。

```
    从上看像山峰
        ↓
    ___/‾\___
   
    从侧面看像山谷
        ↓
    ‾‾‾\___/‾‾‾
```

**特征**：

- 某些方向是最小值
- 其他方向是最大值
- 梯度也为零！

**为什么是问题**：

- SGD会减速甚至停滞
- 即使不是真正的最小值
- **鞍点**：某些方向是最小值，其他方向是最大值

**维度诅咒**：

在高维空间中，鞍点比局部最优更常见！

**直观解释**：

```python
# 1维：只需在1个方向上都是最小值 → 容易
# 2维：需在2个方向上都是最小值 → 较难
# N维：需在N个方向上都是最小值 → 极难

# 鞍点：只需在一些方向是最小值，其他方向是最大值 → 容易多了
```

**研究结果**：

- 对于大型神经网络，几乎所有关键点都是鞍点
- 真正的局部最优非常罕见
- 这实际上是个好消息（鞍点比局部最优容易逃离）
- 高维空间中鞍点更常见

![00:40:34](../../../assets/lecture-3/screenshot-01KG69D0MMGWJ54WCYYX9YDY7V.png) [00:40:34](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:41:43](../../../assets/lecture-3/screenshot-01KG69EXE8M3CDT4A29KZR7VCW.png) [00:41:43](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**问题3：梯度噪声**

**来源**：

- Mini-batch是整个数据集的随机采样
- 每个batch的梯度不完全代表真实梯度
- Mini-batch采样引入噪声
- 更新方向不精确

**数学表示**： $$g_{batch} = g_{true} + noise$$

其中噪声的期望为零，但方差不为零。

**影响**：

```
理想路径:  →→→→→ Minimum

实际路径:  ↗↘→↗→↘→ Minimum
           (抖动但大致正确)
```

**Batch size的权衡**：

```python
# 大batch size
# 噪声小，梯度更准确
# 但每个epoch的更新次数少

# 小batch size
# 噪声大，梯度不准确
# 但每个epoch的更新次数多
# 有时噪声反而有助于逃离坏的局部最优

# 经验法则：
# batch_size ∈ [32, 512]
```

![00:42:25](../../../assets/lecture-3/screenshot-01KG69G33SXWY338HQBTYNS4NZ.png) [00:42:25](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**SGD问题总结**：

| 问题          | 表现               | 根本原因         |
| ------------- | ------------------ | ---------------- |
| 狭窄山谷震荡  | 训练慢，路径之字形 | 各方向曲率差异大 |
| 局部最优/鞍点 | 训练停滞           | 梯度接近零       |
| 梯度噪声      | 路径抖动           | Mini-batch采样   |

**这些问题的解决方案**：

接下来我们将学习的改进算法（Momentum, RMSprop, Adam）正是为了解决这些问题而设计的。

### 5.6 动量法（Momentum）

动量法是对SGD的第一个重要改进，灵感来自物理学。

**物理类比**：

想象一个球滚下山坡：

```
      开始
       ○
        ↘
         ○ 获得速度
          ↘
           ○ 速度越来越快
            ↘
    坑        ○ 冲过小坑
   / \       ↘
  /   \       ○
 /     \___   ↘
            \___○ 最终停在谷底
```

**关键观察**：

- 球会积累动量（momentum）
- 即使遇到小的上坡也能冲过去（逃离局部最优）
- 在一致的方向上会加速

**动机**：像滚动的球一样积累速度

**数学形式**：

**算法**：

```python
v = 0  # 速度初始化

while True:
    dW = compute_gradient(W, X_batch, y_batch)
    v = rho * v - learning_rate * dW  # 更新速度
    W = W + v                          # 更新参数
```

**详细解释**：

1. **维护一个速度变量** $\mathbf{v}$：

   - 初始化为零
   - 在每次迭代中更新
   - 是梯度的指数移动平均

2. **速度更新**： $$\mathbf{v}_{t+1} = \rho \mathbf{v}*t + \nabla*{\mathbf{W}} L(\mathbf{W}_t)$$

   或者（等价形式）： $$\mathbf{v}_{t+1} = \rho \mathbf{v}*t - \alpha \nabla*{\mathbf{W}} L(\mathbf{W}_t)$$

   **解释**：

   - $\rho \mathbf{v}_t$：保留之前的速度（惯性）
   - $\nabla_{\mathbf{W}} L$：当前梯度的贡献
   - $\rho$（通常0.9或0.99）：动量系数

3. **参数更新**： $$\mathbf{W}_{t+1} = \mathbf{W}*t - \alpha \mathbf{v}*{t+1}$$

   或者（另一种形式）： $$\mathbf{W}_{t+1} = \mathbf{W}*t + \mathbf{v}*{t+1}$$

**数学形式**： $$\mathbf{v}_{t+1} = \rho \mathbf{v}*t + \nabla*{\mathbf{W}} L(\mathbf{W}*t)$$ $$\mathbf{W}*{t+1} = \mathbf{W}*t - \alpha \mathbf{v}*{t+1}$$

**参数**：

**$\rho$：动量系数**（通常0.9或0.99）

```python
# ρ = 0：没有动量，退化为标准SGD
v_t = 0.0 * v_{t-1} + grad = grad
W = W - lr * grad  # 就是SGD

# ρ = 0.9：常用值
v_t = 0.9 * v_{t-1} + grad
# 保留90%的历史信息

# ρ = 0.99：强动量
v_t = 0.99 * v_{t-1} + grad
# 保留99%的历史信息，更"惯性"
```

**展开速度项**： $$v_t = \rho v_{t-1} + g_t$$ $$= \rho(\rho v_{t-2} + g_{t-1}) + g_t$$ $$= \rho^2 v_{t-2} + \rho g_{t-1} + g_t$$ $$= g_t + \rho g_{t-1} + \rho^2 g_{t-2} + \rho^3 g_{t-3} + ...$$

**解释**：

- 速度是所有历史梯度的加权和
- 最近的梯度权重最大
- 过去的梯度呈指数衰减
- 高动量 → 更依赖历史方向

**详细实现**：

```python
def sgd_momentum(X, y, learning_rate=0.01, momentum=0.9, 
                 num_epochs=10, batch_size=256):
    """
    带动量的SGD
    """
    N, D = X.shape
    C = np.max(y) + 1
    
    # 初始化
    W = 0.01 * np.random.randn(C, D)
    v = np.zeros_like(W)  # 速度初始化为零
    
    for epoch in range(num_epochs):
        # ... shuffle data ...
        
        for batch_idx in range(num_batches):
            # 获取batch
            X_batch, y_batch = get_batch(...)
            
            # 计算梯度
            dW = compute_gradient(W, X_batch, y_batch)
            
            # 更新速度
            v = momentum * v - learning_rate * dW
            
            # 更新参数
            W += v
    
    return W
```

**优势**：

1. **穿越局部最优和鞍点**

   **机制**：

   ```
   遇到局部最优:
       当前梯度为0
       但速度v不为0（来自之前的动量）
       继续前进！
   
       ○ ← 有动量，继续移动
      /‾\
     /___\  ← 局部最优
   ```

2. **减少狭窄山谷中的震荡**

   **工作原理**：

   ```
   在陡峭方向:
       梯度: ↑↓↑↓ (震荡)
       速度: →→→ (相互抵消后，震荡减小)
   
   在平缓方向:
       梯度: → → → (一致)
       速度: →→→→ (累积加速)
   ```

   **数学解释**：

   ```python
   # 陡峭方向的梯度
   grads_steep = [10, -10, 10, -10, ...]
   
   # 动量使它们相互抵消
   v = 0.9*v + 10    # v ≈ 10
   v = 0.9*v + (-10) # v ≈ 0
   v = 0.9*v + 10    # v ≈ 10
   # 震荡减小！
   
   # 平缓方向的梯度
   grads_flat = [1, 1, 1, 1, ...]
   
   # 动量使它们累积
   v = 0.9*v + 1  # v ≈ 1
   v = 0.9*v + 1  # v ≈ 1.9
   v = 0.9*v + 1  # v ≈ 2.71
   # 加速！
   ```

- 穿越局部最优和鞍点
- 减少狭窄山谷中的震荡

1. **在一致方向上加速**

   如果连续多步梯度方向一致，速度会不断累积。

- 在一致方向上加速

1. **平滑梯度噪声**

   由于是历史梯度的平均，短期噪声被平滑掉了。

- 平滑梯度噪声

**可视化对比**：

```python
# SGD路径
     ↗
    ↗
   ↗  ↘
  ↗    ↘ (之字形，震荡)
 ↗      ↘

# Momentum路径
    →
   →
  → (更直接，更快)
 →
```

![00:42:59](../../../assets/lecture-3/screenshot-01KG69H1MJZWJ9S39EKHAEXQS7.png) [00:42:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**等价形式**：

有两种等价的方式写动量更新：

**形式1**（课程中的）： $$\mathbf{v}_{t+1} = \rho \mathbf{v}*t - \alpha \nabla*{\mathbf{W}} L(\mathbf{W}*t)$$ $$\mathbf{W}*{t+1} = \mathbf{W}*t + \mathbf{v}*{t+1}$$

**形式2**（Nesterov的）： $$\mathbf{v}_{t+1} = \rho \mathbf{v}*t + \nabla*{\mathbf{W}} L(\mathbf{W}*t)$$ $$\mathbf{W}*{t+1} = \mathbf{W}*t - \alpha \mathbf{v}*{t+1}$$

它们在数学上是等价的（可以通过重新定义学习率来证明）。

![00:44:48](../../../assets/lecture-3/screenshot-01KG69M28JCH2JR8Z3ZXJ0478Q.png) [00:44:48](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Nesterov加速梯度（NAG）**：

Momentum的一个改进变体：

```python
# 标准Momentum：
# 1. 计算当前位置的梯度
# 2. 用动量更新

# Nesterov Momentum：
# 1. 先按动量"往前看"
# 2. 在"前方位置"计算梯度
# 3. 用该梯度更新

def nesterov_momentum(W, v, compute_gradient):
    # 往前看
    W_ahead = W + momentum * v
    
    # 在前方计算梯度
    dW = compute_gradient(W_ahead)
    
    # 更新速度
    v = momentum * v - learning_rate * dW
    
    # 更新参数
    W += v
```

**直观理解**：

- Momentum：盲目地跟随惯性
- Nesterov：先看看惯性会带我们去哪，然后做修正
- 通常收敛更快更稳定

**何时使用Momentum**：

- 几乎总是比普通SGD好
- 是Adam等更先进优化器的基础
- 默认值 $\rho=0.9$ 通常效果不错

### 5.7 RMSprop

RMSprop（Root Mean Square Propagation）是另一个重要的优化算法，专注于解决不同方向上学习率应该不同的问题。

**动机**：在不同方向上自适应调整步长

**核心思想**：

- 在陡峭方向上：梯度大 → 需要小步长（避免震荡）
- 在平缓方向上：梯度小 → 需要大步长（加速收敛）

**算法**：

```python
grad_squared = 0

while True:
    dW = compute_gradient(W, X_batch, y_batch)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dW**2
    W = W - learning_rate * dW / (np.sqrt(grad_squared) + 1e-7)
```

**数学形式**：

1. **累积梯度平方的移动平均**： $$\mathbf{s}*t = \beta \mathbf{s}*{t-1} + (1-\beta) (\nabla_{\mathbf{W}} L)^2$$

   **注意**：

   - 这是**逐元素**的平方（不是矩阵的平方）
   - $\mathbf{s}_t$ 与 $\mathbf{W}$ 形状相同
   - $\beta$：衰减率（通常0.9或0.99）

2. **自适应参数更新**： $$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\mathbf{s}*t} + \epsilon} \nabla*{\mathbf{W}} L$$

   **注意**：

   - 除法是逐元素的
   - $\epsilon$（如1e-8）防止除以零
   - $\sqrt{\mathbf{s}_t}$ 是RMS（均方根）

**详细工作原理**：

**场景1：方向上梯度一直很大（陡峭方向）**

```python
# 梯度序列
grads = [10, 10, 10, 10, ...]

# s的演化（β=0.9）
s_1 = 0.9*0 + 0.1*10² = 10
s_2 = 0.9*10 + 0.1*10² = 19
s_3 = 0.9*19 + 0.1*10² = 27.1
...
s_∞ ≈ 100  # 稳定值

# 有效学习率
lr_effective = lr / sqrt(100) = lr / 10
# 步长被缩小10倍！
```

**场景2：方向上梯度一直很小（平缓方向）**

```python
# 梯度序列
grads = [0.1, 0.1, 0.1, 0.1, ...]

# s的演化
s_1 = 0.9*0 + 0.1*0.1² = 0.001
s_2 = 0.9*0.001 + 0.1*0.1² = 0.0019
...
s_∞ ≈ 0.01

# 有效学习率
lr_effective = lr / sqrt(0.01) = lr / 0.1 = 10*lr
# 步长被放大10倍！
```

**效果**：

- **在陡峭方向上减小步长**（除以大的梯度平方）
- **在平缓方向上增大步长**（除以小的梯度平方）

**详细实现**：

```python
def rmsprop(X, y, learning_rate=0.001, decay_rate=0.9,
            num_epochs=10, batch_size=256, epsilon=1e-8):
    """
    RMSprop优化器
    """
    N, D = X.shape
    C = np.max(y) + 1
    
    # 初始化
    W = 0.01 * np.random.randn(C, D)
    cache = np.zeros_like(W)  # 梯度平方的累积
    
    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            # 获取batch和计算梯度
            X_batch, y_batch = get_batch(...)
            dW = compute_gradient(W, X_batch, y_batch)
            
            # 更新梯度平方的缓存
            cache = decay_rate * cache + (1 - decay_rate) * (dW ** 2)
            
            # 自适应学习率更新
            W -= learning_rate * dW / (np.sqrt(cache) + epsilon)
    
    return W
```

- 自动适应损失景观的几何形状

**为什么叫RMSprop**：

RMS = Root Mean Square（均方根）

$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum x_i^2}$$

在我们的算法中： $$\sqrt{\mathbf{s}*t} = \sqrt{\beta \mathbf{s}*{t-1} + (1-\beta)g_t^2}$$

这近似于梯度的RMS。

**与AdaGrad的关系**：

RMSprop是AdaGrad的改进版本。

**AdaGrad**： $$\mathbf{s}*t = \mathbf{s}*{t-1} + g_t^2$$  （累加所有历史梯度平方）

**问题**：

- $\mathbf{s}_t$ 单调递增
- 学习率单调递减
- 最终可能变为零，训练停止

**RMSprop的改进**： $$\mathbf{s}*t = \beta \mathbf{s}*{t-1} + (1-\beta)g_t^2$$  （指数移动平均）

**优点**：

- 更重视最近的梯度
- 学习率可以增大或减小
- 不会过早停止训练

![00:49:38](../../../assets/lecture-3/screenshot-01KG69W31EBN5B37NPFQ38KMXK.png) [00:49:38](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**可视化**：

在狭窄山谷的例子中：

```
陡峭方向(y):
    梯度大 → s_y大 → 除以大数 → 步长小 ✓

平缓方向(x):
    梯度小 → s_x小 → 除以小数 → 步长大 ✓

结果：
    更直接地朝向最小值前进
```

![00:51:36](../../../assets/lecture-3/screenshot-01KG69ZC99CZ6VW69WB4ZEN429.png) [00:51:36](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**何时使用RMSprop**：

- 当你的损失函数有很不同的曲率时
- RNN训练中特别有效
- 通常比AdaGrad好

![00:53:06](../../../assets/lecture-3/screenshot-01KG6A1VGQH9TWTZW66ADZH39K.png) [00:53:06](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.8 Adam优化器（最流行）

Adam（Adaptive Moment Estimation）是目前最流行的优化算法，它巧妙地结合了Momentum和RMSprop的优点。

**Adam = Momentum + RMSprop**

**核心思想**：

- 像Momentum一样：维护梯度的移动平均（一阶矩）
- 像RMSprop一样：维护梯度平方的移动平均（二阶矩）
- 额外改进：偏差修正

**算法**：

```python
m = 0  # 一阶矩估计（动量）
v = 0  # 二阶矩估计（RMSprop）
t = 0  # 时间步

while True:
    t += 1
    dW = compute_gradient(W, X_batch, y_batch)
    
    # 更新一阶和二阶矩估计
    m = beta1 * m + (1 - beta1) * dW
    v = beta2 * v + (1 - beta2) * (dW**2)
    
    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # 参数更新
    W = W - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
```

**详细数学形式**：

1. **一阶矩（Momentum部分）**： $$\mathbf{m}*t = \beta_1 \mathbf{m}*{t-1} + (1-\beta_1) \nabla_{\mathbf{W}} L$$
   - 类似Momentum的速度
   - $\beta_1$通常取0.9
2. **二阶矩（RMSprop部分）**： $$\mathbf{v}*t = \beta_2 \mathbf{v}*{t-1} + (1-\beta_2) (\nabla_{\mathbf{W}} L)^2$$
   - 类似RMSprop的缓存
   - $\beta_2$通常取0.999
3. **偏差修正**： $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
   - 这是Adam的创新
   - 解决初始化偏差问题（见下文）
4. **参数更新**： $$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

**推荐超参数**：

基于大量实验，这些默认值在大多数问题上效果不错：

- $\beta_1 = 0.9$ （一阶矩的衰减率）
- $\beta_2 = 0.999$ （二阶矩的衰减率）
- $\alpha = 1e-3$ 或 $5e-4$ （学习率）
- $\epsilon = 1e-8$ （数值稳定性）

**偏差修正的必要性**：

这是Adam相对于简单组合Momentum和RMSprop的关键改进。

**问题**：

```python
# 初始化
m_0 = 0
v_0 = 0

# 第一次更新（假设梯度g=1，β₁=0.9）
m_1 = 0.9 * 0 + 0.1 * 1 = 0.1
# 实际期望应该是1，但我们得到0.1
# 严重偏向零！
```

**数学分析**：

如果我们展开 $\mathbf{m}_t$： $$\mathbf{m}*t = (1-\beta_1)\sum*{i=1}^{t} \beta_1^{t-i} g_i$$

当 $t$ 很小时，$\sum_{i=1}^{t} \beta_1^{t-i} < \frac{1}{1-\beta_1}$

所以 $\mathbf{m}_t$ 被低估了。

**解决方案**：偏差修正

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}$$

**验证**：

```python
# t=1时
m_1 = 0.1
correction = 1 - 0.9^1 = 0.1
m_hat_1 = 0.1 / 0.1 = 1.0  ✓

# t=2时
m_2 = 0.9*0.1 + 0.1*1 = 0.19
correction = 1 - 0.9^2 = 0.19
m_hat_2 = 0.19 / 0.19 = 1.0  ✓

# t→∞时
# 1 - β₁^t → 1
# 偏差修正变为除以1，即无影响
```

- $\mathbf{m}_0 = \mathbf{v}_0 = 0$（初始化为零）
- 早期时间步，矩估计偏向零
- 偏差修正解决初期步长过大的问题

**完整实现**：

```python
def adam(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999,
         num_epochs=10, batch_size=256, epsilon=1e-8):
    """
    Adam优化器
    """
    N, D = X.shape
    C = np.max(y) + 1
    
    # 初始化
    W = 0.01 * np.random.randn(C, D)
    m = np.zeros_like(W)  # 一阶矩
    v = np.zeros_like(W)  # 二阶矩
    t = 0  # 时间步
    
    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            t += 1  # 递增时间步
            
            # 计算梯度
            X_batch, y_batch = get_batch(...)
            dW = compute_gradient(W, X_batch, y_batch)
            
            # 更新一阶矩
            m = beta1 * m + (1 - beta1) * dW
            
            # 更新二阶矩
            v = beta2 * v + (1 - beta2) * (dW ** 2)
            
            # 偏差修正
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # 参数更新
            W -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return W
```

![00:53:50](../../../assets/lecture-3/screenshot-01KG6A32W69B5Q25E9ARW803XQ.png) [00:53:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Adam的优点**：

1. **结合了Momentum和RMSprop的优点**
   - 动量：加速收敛，穿越鞍点
   - 自适应学习率：处理不同曲率
2. **超参数鲁棒性**
   - 默认参数在大多数问题上效果不错
   - 不需要大量调参
3. **偏差修正**
   - 解决了初期更新的问题
   - 训练初期更稳定
4. **广泛适用**
   - 在各种深度学习任务上效果好
   - CNN、RNN、Transformer等

![00:56:09](../../../assets/lecture-3/screenshot-01KG6A6YRHSVA1ZKGE9PPGTV2A.png) [00:56:09](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**Adam的局限性**：

1. **可能收敛到较差的局部最优**
   - 有研究表明，在某些情况下SGD+Momentum泛化更好
   - 但Adam收敛更快
2. **学习率调度仍然重要**
   - 虽然自适应，但仍建议使用学习率衰减
3. **内存占用**
   - 需要存储m和v（是参数的2倍内存）

![00:57:05](../../../assets/lecture-3/screenshot-01KG6A8GFAKTWKYG991SJN82HA.png) [00:57:05](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.9 AdamW（Adam with Decoupled Weight Decay）

AdamW是Adam的改进版本，主要改进了权重衰减（L2正则化）的处理方式。

**关键区别**：将权重衰减与梯度解耦

**动机**：

在标准Adam中，L2正则化是这样处理的：

**标准Adam**：

```python
# 计算梯度（包含L2正则化）
dW = compute_gradient(W) + lambda * W  # L2正则项包含在梯度中

# 然后用Adam更新
m = beta1 * m + (1 - beta1) * dW
v = beta2 * v + (1 - beta2) * dW**2
...
```

**问题**：

- L2正则项 `lambda * W` 被包含在梯度中
- 它会影响动量m和二阶矩v的计算
- 这导致正则化的效果与优化器耦合

**AdamW的改进**：

**AdamW**：

```python
# 计算梯度（不包含正则项）
dW = compute_gradient(W)  # 不包含正则项

# Adam更新（不涉及正则化）
m = beta1 * m + (1 - beta1) * dW
v = beta2 * v + (1 - beta2) * dW**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)

# 参数更新 + 单独应用权重衰减
W = W - lr * m_hat / sqrt(v_hat) - lr * lambda * W  # 单独应用权重衰减
```

**数学形式**：

**标准Adam + L2**： $$\mathbf{W}*t = \mathbf{W}*{t-1} - \alpha_t \frac{\hat{\mathbf{m}}*t}{\sqrt{\hat{\mathbf{v}}\*t} + \epsilon}$$ 其中梯度 $g_t = \nabla L\*{data} + \lambda \mathbf{W}*{t-1}$

**AdamW**： $$\mathbf{W}*t = \mathbf{W}*{t-1} - \alpha_t \frac{\hat{\mathbf{m}}*t}{\sqrt{\hat{\mathbf{v}}\*t} + \epsilon} - \alpha_t \lambda \mathbf{W}\*{t-1}$$ 其中梯度 $g_t = \nabla L*{data}$（不包含正则项）

**实现**：

```python
def adamw(X, y, learning_rate=0.001, weight_decay=0.01,
          beta1=0.9, beta2=0.999, num_epochs=10):
    """
    AdamW优化器
    """
    W = 0.01 * np.random.randn(C, D)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    t = 0
    
    for epoch in range(num_epochs):
        for batch in batches:
            t += 1
            
            # 计算梯度（不包含正则化）
            dW = compute_gradient_without_reg(W, batch)
            
            # Adam部分（标准）
            m = beta1 * m + (1 - beta1) * dW
            v = beta2 * v + (1 - beta2) * dW**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # 更新：Adam + 解耦的权重衰减
            W = W - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            W = W - learning_rate * weight_decay * W  # 权重衰减单独应用
    
    return W
```

**优势**：

1. **更好地分离优化和正则化**
   - 权重衰减不影响动量计算
   - 正则化效果更可预测
   - 超参数调节更独立
2. **更好的泛化性能**
   - 在许多任务上表现优于标准Adam
   - 特别是在Transformer模型中

- 权重衰减不影响动量计算
- 更好地分离优化和正则化

1. **与SGD+Momentum的权重衰减一致**
   - SGD中：`W = W - lr*grad - lr*lambda*W`
   - AdamW保持了这种形式

- 在许多任务上表现优于Adam（如Llama系列模型）

**何时使用AdamW**：

- 训练Transformer模型（几乎是标准选择）
- 当你需要强正则化时
- 大多数现代深度学习任务

**在PyTorch中**：

```python
import torch.optim as optim

# 使用AdamW
optimizer = optim.AdamW(model.parameters(), 
                        lr=1e-3, 
                        weight_decay=0.01)
```

![00:57:22](../../../assets/lecture-3/screenshot-01KG6A8ZACN405C8R50JXTD0MG.png) [00:57:22](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**对比总结**：

| 方法      | 权重衰减方式 | 与优化器的关系 |
| --------- | ------------ | -------------- |
| Adam + L2 | 包含在梯度中 | 耦合           |
| AdamW     | 单独应用     | 解耦           |

![00:58:27](../../../assets/lecture-3/screenshot-01KG6AARRKR6MZE8JSJWMKGZ8P.png) [00:58:27](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

------

## 6. 学习率调度

即使有了好的优化器（如Adam），学习率的选择和调整仍然至关重要。学习率调度（Learning Rate Scheduling）是指在训练过程中动态调整学习率的策略。

### 6.1 学习率的重要性

学习率可能是最重要的超参数。

**学习率过高**：

```python
# 学习率 = 10.0（太大）
迭代1: loss = 2.5
迭代2: loss = 15.3  # 爆炸！
迭代3: loss = 247.8  # 完全发散
迭代4: loss = NaN  # 数值溢出
```

表现：

- **损失值爆炸**：迅速增长到无穷大
- **在最优点附近震荡**：永远无法收敛
- **无法收敛**：训练完全失败

**可视化**：

```
      损失景观
    ___/ \___
   步子太大，跳过了最小值
     ↓
   ←─────→  (跳来跳去)
```

**学习率过低**：

```python
# 学习率 = 0.000001（太小）
迭代1: loss = 2.5
迭代100: loss = 2.499
迭代1000: loss = 2.49
迭代10000: loss = 2.4
# 100小时后...
```

表现：

- **收敛速度极慢**：需要非常多迭代
- **可能卡在鞍点**：梯度很小时无法逃离
- **训练时间过长**：不实用

**可视化**：

```
      损失景观
    ___/ \___
   步子太小，慢慢爬
     ↓
   . . . . (一点一点移动)
```

**理想学习率**：

```python
# 学习率 = 0.001（刚好）
迭代1: loss = 2.5
迭代10: loss = 2.0
迭代50: loss = 1.0
迭代100: loss = 0.3
迭代200: loss = 0.15  # 稳定收敛
```

表现：

- **快速下降**：损失迅速降低
- **稳定收敛到较低损失**：最终达到好的结果
- **不震荡**：路径平滑

![00:59:30](../../../assets/lecture-3/screenshot-01KG6ACGSZX3F57BHR2AW90MSH.png) [00:59:30](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**如何找到合适的学习率**：

**方法1：网格搜索**

```python
# 在对数尺度上搜索
lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

for lr in lrs:
    model = train(lr=lr, num_epochs=5)
    val_loss = evaluate(model)
    print(f"LR {lr}: Val Loss = {val_loss}")

# 选择验证损失最低的
```

**方法2：学习率范围测试（LR Range Test）**

```python
# 从很小的学习率开始，逐渐增大
lr_min = 1e-
```