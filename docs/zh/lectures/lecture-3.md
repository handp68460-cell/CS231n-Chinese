# Lecture 2: Regularization and Optimization

## 课程概述

本节课深入探讨了深度学习和计算机视觉中的两个核心概念：**正则化（Regularization）**和**优化（Optimization）**。课程首先回顾了图像分类任务和线性分类器的基本原理，然后详细讲解了如何通过损失函数评估模型性能，如何使用正则化防止过拟合，以及如何通过各种优化算法（特别是梯度下降及其变体）来训练模型。

## 目录

1. [回顾：图像分类与线性分类器](#1-回顾图像分类与线性分类器)
2. [损失函数](#2-损失函数)
3. [正则化（Regularization）](#3-正则化regularization)
4. [优化（Optimization）](#4-优化optimization)
5. [梯度下降及其变体](#5-梯度下降及其变体)
6. [学习率调度](#6-学习率调度)
7. [总结与展望](#7-总结与展望)

---

## 1. 回顾：图像分类与线性分类器

### 1.1 图像分类任务

**图像分类**是计算机视觉的核心任务之一，目标是将输入图像映射到预定义的类别标签集合中。

**任务定义**：
- **输入**：图像（像素值的多维数组）
- **输出**：类别标签（如：猫、狗、鸟、鹿、卡车）

### 1.2 图像分类的挑战

1. **语义鸿沟（Semantic Gap）**
   - 人类感知：看到"猫"
   - 计算机表示：像素值的数值网格
   
2. **图像变化**
   - **光照变化**：不同光照条件下像素强度不同
   - **遮挡（Occlusion）**：物体部分被遮挡
   - **形变（Deformation）**：物体可以弯曲、扭转
   - **背景杂波**：物体与背景融合
   - **类内变化（Intra-class Variation）**：同一类别的不同实例外观差异大



![00:00:56](../../assets/lecture-3/screenshot-01KG66KKM3RNW3T52AGRKH04YJ.png)
[00:00:56](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 1.3 K近邻分类器（K-Nearest Neighbors）

**基本思想**：对于新的数据点，找到训练集中距离最近的K个样本，采用多数投票决定类别。

![00:02:55](../../assets/lecture-3/screenshot-01KG66PQV6V9DJBX2KCZJRFZHG.png)
[00:02:55](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

*距离度量**：

1. **L1距离（曼哈顿距离）**：
   $$d_{L1}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|$$

2. **L2距离（欧氏距离）**：
   $$d_{L2}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**超参数选择**：
- 使用训练集/验证集/测试集划分
- 在验证集上选择最优的K值
- 在测试集上评估最终性能

![00:04:22](../../assets/lecture-3/screenshot-01KG66S50V77HW9AX1Z199PA2E.png)
[00:04:22](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:05:05](../../assets/lecture-3/screenshot-01KG66TAXWW1G45HEVDKZWWVZ4.png)
[00:05:05](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 1.4 线性分类器

**模型公式**：
$$f(\mathbf{x}, \mathbf{W}) = \mathbf{W}\mathbf{x} + \mathbf{b}$$

其中：
- $\mathbf{x}$：输入图像（展平为向量，大小为 $32 \times 32 \times 3 = 3072$）
- $\mathbf{W}$：权重矩阵（大小为 $C \times D$，C为类别数，D为特征维度）
- $\mathbf{b}$：偏置向量（大小为 $C$）

**三种理解视角**：

1. **代数视角**：每一行权重与输入向量做点积，得到该类别的得分

2. **模板视角**：每一行权重可视化为该类别的"模板"

3. **几何视角**：每一行权重定义了特征空间中的决策边界（超平面）

**局限性**：线性分类器只能学习线性决策边界，无法处理非线性可分的数据

---

![00:06:31](../../assets/lecture-3/screenshot-01KG66WQNAH1SN6ZN1RPJBYGB0.png)
[00:06:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 2. 损失函数

### 2.1 损失函数的定义

**损失函数（Loss Function）** 量化了模型预测与真实标签之间的差距。

**数学定义**：
$$L = \frac{1}{N} \sum_{i=1}^{N} L_i(f(\mathbf{x}_i, \mathbf{W}), y_i) + \lambda R(\mathbf{W})$$

其中：
- **数据损失（Data Loss）**：$\frac{1}{N} \sum_{i=1}^{N} L_i$ - 衡量模型对训练数据的拟合程度
- **正则化损失（Regularization Loss）**：$\lambda R(\mathbf{W})$ - 防止过拟合
- $\lambda$：正则化强度（超参数）

### 2.2 Softmax损失（交叉熵损失）

**Softmax函数**：将任意实数向量转换为概率分布
$$P(y_i = k | \mathbf{x}_i) = \frac{e^{s_k}}{\sum_{j} e^{s_j}}$$

**交叉熵损失**：
$$L_i = -\log P(y_i = y_{true} | \mathbf{x}_i) = -\log \left(\frac{e^{s_{y_{true}}}}{\sum_{j} e^{s_j}}\right)$$

**特性**：
- 当正确类别概率高时，损失低
- 当正确类别概率低时，损失高
- 鼓励模型输出高置信度的正确预测

---

## 3. 正则化（Regularization）

### 3.1 正则化的动机

**核心思想**：在训练数据上表现稍差，但在测试数据上表现更好。

**直观例子**：
- 模型F1：完美拟合所有训练点（可能过拟合）
- 模型F2：更简单的模型，训练误差稍高，但泛化能力更强



![00:11:54](../../assets/lecture-3/screenshot-01KG675NF37KTXS94GJ9746GSC.png)
[00:11:54](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:13:04](../../assets/lecture-3/screenshot-01KG677M1DR49XQD000PQ2CRXR.png)
[00:13:04](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:13:18](../../assets/lecture-3/screenshot-01KG6780SP9A2Z9NN945RGWP7M.png)
[00:13:18](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**奥卡姆剃刀原则**：在多个竞争假设中，优先选择最简单的。

![00:14:19](../../assets/lecture-3/screenshot-01KG679PRKFRZ9HY5QECY72TR8.png)
[00:14:19](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:14:59](../../assets/lecture-3/screenshot-01KG67ATG66Q150QV9PTGT6X96.png)
[00:14:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 3.2 L2正则化（权重衰减）

**定义**：
$$R(\mathbf{W}) = \sum_{i,j} W_{i,j}^2$$

**特性**：
- 惩罚大的权重值
- 偏好**分散的**权重分布
- 由于平方操作，小权重值的惩罚更小
- 导致权重值普遍较小但非零

**梯度**：
$$\frac{\partial R}{\partial W_{i,j}} = 2W_{i,j}$$

### 3.3 L1正则化

**定义**：
$$R(\mathbf{W}) = \sum_{i,j} |W_{i,j}|$$

**特性**：
- 偏好**稀疏的**权重分布
- 导致许多权重值精确为0
- 可用于特征选择

**梯度**：
$$\frac{\partial R}{\partial W_{i,j}} = \text{sign}(W_{i,j})$$

![00:15:29](../../assets/lecture-3/screenshot-01KG67BN53BS2QQV576H3ZN8M8.png)
[00:15:29](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:18:21](../../assets/lecture-3/screenshot-01KG67GDZCW7FGZZRZR9332PPQ.png)
[00:18:21](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:18:31](../../assets/lecture-3/screenshot-01KG67GPKKWCNQTSZ6HWD8KQ2C.png)
[00:18:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 3.4 L1 vs L2 对比示例

考虑两组权重，得分相同：
- $\mathbf{w}_1 = [1, 0, 0, 0]$
- $\mathbf{w}_2 = [0.25, 0.25, 0.25, 0.25]$

**L2正则化**：
- $R(\mathbf{w}_1) = 1^2 = 1$
- $R(\mathbf{w}_2) = 4 \times (0.25)^2 = 0.25$
- **L2偏好** $\mathbf{w}_2$（更分散）

**L1正则化**：
- $R(\mathbf{w}_1) = 1$
- $R(\mathbf{w}_2) = 4 \times 0.25 = 1$
- 两者相同（但实践中L1会倾向于稀疏解）

![00:20:20](../../assets/lecture-3/screenshot-01KG67KQ66ZM7JVNXG7A22Y6RN.png)
[00:20:20](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:20:37](../../assets/lecture-3/screenshot-01KG67M6CTXCM5M5XEH541D7RW.png)
[00:20:37](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:21:43](../../assets/lecture-3/screenshot-01KG67P6NH5SF85EREJ35E3PJE.png)
[00:21:43](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:26:32](../../assets/lecture-3/screenshot-01KG67Z1NHSN1QBJT7X41KX68R.png)
[00:26:32](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 3.5 正则化的作用

1. **表达模型偏好**：通过选择不同的正则化项表达对权重的偏好
2. **提高泛化能力**：简化模型，防止过拟合
3. **改善优化**：L2正则化的凸性有助于优化过程

---



## 4. 优化（Optimization）

### 4.1 优化问题

**目标**：找到使损失函数最小的权重 $\mathbf{W}^*$
$$\mathbf{W}^* = \arg\min_{\mathbf{W}} L(\mathbf{W})$$

### 4.2 损失景观（Loss Landscape）

**可视化比喻**：
- **垂直轴（Z轴）**：损失值
- **水平轴（X, Y轴）**：模型参数
- **目标**：找到景观中的最低点

**关键限制**：
- 我们是"蒙眼"的 - 只能感知当前位置的局部梯度
- 无法直接看到全局最优点

![00:26:54](../../assets/lecture-3/screenshot-01KG67ZPE5KHBCTZE27090Z22G.png)
[00:26:54](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 4.3 策略1：随机搜索（不推荐）

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

**性能**：CIFAR-10上约15.5%准确率（远低于99.7%的最优水平）

### 4.4 策略2：跟随斜率（梯度下降）

**核心思想**：计算当前位置的斜率，朝下坡方向移动



## 5. 梯度下降及其变体

### 5.1 梯度的数学定义

**一维导数**：
$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**多维梯度**：
$$\nabla_{\mathbf{W}} L = \left[\frac{\partial L}{\partial W_{1,1}}, \frac{\partial L}{\partial W_{1,2}}, \ldots, \frac{\partial L}{\partial W_{m,n}}\right]$$

**重要性质**：

- 梯度方向是函数**上升最快**的方向
- 负梯度方向是函数**下降最快**的方向

### 5.2 梯度计算方法

**方法1：数值梯度（慢，近似）**
```python
def numerical_gradient(f, W, h=1e-5):
    grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] += h
            fpos = f(W)
            W[i,j] -= 2*h
            fneg = f(W)
            W[i,j] += h  # 恢复
            grad[i,j] = (fpos - fneg) / (2*h)
    return grad
```

**方法2：解析梯度（快，精确）**
- 使用微积分链式法则推导
- 精确且高效
- **推荐使用，并用数值梯度验证**



![00:27:33](../../assets/lecture-3/screenshot-01KG680SA9YBHPMA9X9KTT55H9.png)
[00:27:33](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:29:35](../../assets/lecture-3/screenshot-01KG6845SW06B0WBAH5TRKMFH2.png)
[00:29:35](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:30:11](../../assets/lecture-3/screenshot-01KG6855GY1MW2J0VPMG4GHK5V.png)
[00:30:11](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:30:50](../../assets/lecture-3/screenshot-01KG68R314QGSSQK2K6J1P2Y0P.png)
[00:30:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:31:35](../../assets/lecture-3/screenshot-01KG68SBKB1SCA92ZDK28JSC0S.png)
[00:31:35](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)



### 5.3 梯度下降算法

**基本算法**：
```python
# 伪代码
while True:
    dW = compute_gradient(W, X, y)  # 计算梯度
    W = W - learning_rate * dW       # 参数更新
```

**数学形式**：
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \nabla_{\mathbf{W}} L(\mathbf{W}_t)$$

其中 $\alpha$ 是**学习率（步长）**

![00:34:01](../../assets/lecture-3/screenshot-01KG68Y89T430EGN51KPAKGRCK.png)
[00:34:01](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:34:55](../../assets/lecture-3/screenshot-01KG68ZRQC1FAJ970JT9GFPPH6.png)
[00:34:55](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:35:34](../../assets/lecture-3/screenshot-01KG690WN7WAXKJ19SQN1QATBR.png)
[00:35:34](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.4 随机梯度下降（SGD）

**动机**：计算全部数据的梯度开销太大

**Mini-batch SGD**：
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

**关键概念**：
- **Batch Size**：每次更新使用的样本数（如256）
- **Epoch**：遍历整个训练集一次
- **优势**：计算高效，引入随机性有助于逃离局部最优

---

![00:36:58](../../assets/lecture-3/screenshot-01KG69713PWV5GB4CY757R4NGJ.png)
[00:36:58](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:37:58](../../assets/lecture-3/screenshot-01KG698NM3RMF42WBNCCFFQT3W.png)
[00:37:58](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.5 SGD的问题

**问题1：狭窄山谷中的震荡**
- 在陡峭方向上大幅震荡
- 在平缓方向上进展缓慢
- **原因**：条件数（condition number）高 - Hessian矩阵最大/最小特征值比值大



![00:39:13](../../assets/lecture-3/screenshot-01KG69ARKRKEGYDVBKZ6D7E6R4.png)
[00:39:13](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)



**问题2：局部最优和鞍点**

- **局部最优**：梯度为零但不是全局最优
- **鞍点**：某些方向是最小值，其他方向是最大值
- 高维空间中鞍点更常见

![00:40:34](../../assets/lecture-3/screenshot-01KG69D0MMGWJ54WCYYX9YDY7V.png)
[00:40:34](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:41:43](../../assets/lecture-3/screenshot-01KG69EXE8M3CDT4A29KZR7VCW.png)
[00:41:43](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

**问题3：梯度噪声**

- Mini-batch采样引入噪声
- 更新方向不精确

![00:42:25](../../assets/lecture-3/screenshot-01KG69G33SXWY338HQBTYNS4NZ.png)
[00:42:25](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.6 动量法（Momentum）

**动机**：像滚动的球一样积累速度

**算法**：
```python
v = 0  # 速度初始化

while True:
    dW = compute_gradient(W, X_batch, y_batch)
    v = rho * v - learning_rate * dW  # 更新速度
    W = W + v                          # 更新参数
```

**数学形式**：
$$\mathbf{v}_{t+1} = \rho \mathbf{v}_t + \nabla_{\mathbf{W}} L(\mathbf{W}_t)$$
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \mathbf{v}_{t+1}$$

**参数**：
- $\rho$：动量系数（通常0.9或0.99）
- 高动量 → 更依赖历史方向

**优势**：
- 穿越局部最优和鞍点
- 减少狭窄山谷中的震荡
- 在一致方向上加速
- 平滑梯度噪声

**等价形式**：
$$\mathbf{v}_{t+1} = \rho \mathbf{v}_t - \alpha \nabla_{\mathbf{W}} L(\mathbf{W}_t)$$
$$\mathbf{W}_{t+1} = \mathbf{W}_t + \mathbf{v}_{t+1}$$

![00:42:59](../../assets/lecture-3/screenshot-01KG69H1MJZWJ9S39EKHAEXQS7.png)
[00:42:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:44:48](../../assets/lecture-3/screenshot-01KG69M28JCH2JR8Z3ZXJ0478Q.png)
[00:44:48](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.7 RMSprop

**动机**：在不同方向上自适应调整步长

**算法**：
```python
grad_squared = 0

while True:
    dW = compute_gradient(W, X_batch, y_batch)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dW**2
    W = W - learning_rate * dW / (np.sqrt(grad_squared) + 1e-7)
```

**数学形式**：
$$\mathbf{s}_t = \beta \mathbf{s}_{t-1} + (1-\beta) (\nabla_{\mathbf{W}} L)^2$$
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\mathbf{s}_t} + \epsilon} \nabla_{\mathbf{W}} L$$

**效果**：
- 在陡峭方向上减小步长（除以大的梯度平方）
- 在平缓方向上增大步长（除以小的梯度平方）
- 自动适应损失景观的几何形状

![00:49:38](../../assets/lecture-3/screenshot-01KG69W31EBN5B37NPFQ38KMXK.png)
[00:49:38](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:51:36](../../assets/lecture-3/screenshot-01KG69ZC99CZ6VW69WB4ZEN429.png)
[00:51:36](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:53:06](../../assets/lecture-3/screenshot-01KG6A1VGQH9TWTZW66ADZH39K.png)
[00:53:06](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.8 Adam优化器（最流行）

**Adam = Momentum + RMSprop**

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

**数学形式**：
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_{\mathbf{W}} L$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_{\mathbf{W}} L)^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

**推荐超参数**：
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\alpha = 1e-3$ 或 $5e-4$
- $\epsilon = 1e-8$

**偏差修正的必要性**：
- $\mathbf{m}_0 = \mathbf{v}_0 = 0$（初始化为零）
- 早期时间步，矩估计偏向零
- 偏差修正解决初期步长过大的问题

![00:53:50](../../assets/lecture-3/screenshot-01KG6A32W69B5Q25E9ARW803XQ.png)
[00:53:50](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:56:09](../../assets/lecture-3/screenshot-01KG6A6YRHSVA1ZKGE9PPGTV2A.png)
[00:56:09](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:57:05](../../assets/lecture-3/screenshot-01KG6A8GFAKTWKYG991SJN82HA.png)
[00:57:05](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 5.9 AdamW（Adam with Decoupled Weight Decay）

**关键区别**：将权重衰减与梯度解耦

**标准Adam**：
```python
dW = compute_gradient(W) + lambda * W  # L2正则项包含在梯度中
```

**AdamW**：
```python
dW = compute_gradient(W)  # 不包含正则项
W = W - lr * m / sqrt(v) - lr * lambda * W  # 单独应用权重衰减
```

**优势**：
- 权重衰减不影响动量计算
- 更好地分离优化和正则化
- 在许多任务上表现优于Adam（如Llama系列模型）

---

![00:57:22](../../assets/lecture-3/screenshot-01KG6A8ZACN405C8R50JXTD0MG.png)
[00:57:22](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![00:58:27](../../assets/lecture-3/screenshot-01KG6AARRKR6MZE8JSJWMKGZ8P.png)
[00:58:27](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 6. 学习率调度

### 6.1 学习率的重要性

**学习率过高**：
- 损失值爆炸
- 在最优点附近震荡
- 无法收敛

**学习率过低**：
- 收敛速度极慢
- 可能卡在鞍点

**理想学习率**：
- 快速下降
- 稳定收敛到较低损失

![00:59:30](../../assets/lecture-3/screenshot-01KG6ACGSZX3F57BHR2AW90MSH.png)
[00:59:30](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:00:04](../../assets/lecture-3/screenshot-01KG6ADF824EYP86NE5HJV9CER.png)
[01:00:04](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 6.2 学习率衰减策略

**1. 阶梯衰减（Step Decay）**
```python
if epoch % step_size == 0:
    learning_rate *= gamma  # 如 gamma=0.1
```
- 每隔固定epoch减小学习率
- ResNet训练中常用

**2. 余弦退火（Cosine Annealing）**
$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\frac{t\pi}{T}))$$
```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / max_epoch * np.pi))
```
- 平滑减小学习率
- 非常流行

**3. 线性衰减（Linear Decay）**
$$\alpha_t = \alpha_0 (1 - \frac{t}{T})$$

**4. 逆平方根衰减（Inverse Square Root）**
$$\alpha_t = \frac{\alpha_0}{\sqrt{t}}$$

![01:00:44](../../assets/lecture-3/screenshot-01KG6AG71N1NBJNHWSEFW7R8BG.png)
[01:00:44](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 6.3 学习率预热（Warmup）

**策略**：训练初期线性增加学习率

```python
if epoch < warmup_epochs:
    lr = lr_max * epoch / warmup_epochs
else:
    lr = cosine_decay(epoch - warmup_epochs)
```

**常见组合**：
- Linear Warmup + Cosine Decay
- Linear Warmup + Step Decay

**优势**：
- 避免初期大步长导致的不稳定
- 在大batch size训练中尤其重要

![01:01:06](../../assets/lecture-3/screenshot-01KG6AGT1KFRX6Q12TS6JSP1KZ.png)
[01:01:06](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 6.4 线性缩放规则（Linear Scaling Rule）

**经验法则**：
$$\text{学习率} \propto \text{Batch Size}$$

**示例**：
- Batch size从256增加到512
- 学习率也应从0.1增加到0.2

**理论依据**：
- Batch size越大，梯度估计越准确
- 可以使用更大的步长

![01:01:24](../../assets/lecture-3/screenshot-01KG6AHA2QB69QEKP4YJBFQ8A2.png)
[01:01:24](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:01:48](../../assets/lecture-3/screenshot-01KG6AHZE3SMEBQBQPHS7WCM1V.png)
[01:01:48](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:02:33](../../assets/lecture-3/screenshot-01KG6AQZCDA8Q7BR9CA085D1WM.png)
[01:02:33](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:03:47](../../assets/lecture-3/screenshot-01KG6AT17S7ZNRGY2R6H4RZSV7.png)
[01:03:47](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:04:28](../../assets/lecture-3/screenshot-01KG6AV5PMX697NN9TAYGPNGAG.png)
[01:04:28](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

## 7. 总结与展望

### 7.1 核心要点总结

**正则化**：
- 在训练数据上表现稍差，在测试数据上表现更好
- L2正则化偏好分散权重，L1正则化偏好稀疏权重
- 权重衰减是一种正则化策略

**优化算法**：
- **SGD**：简单但可能震荡、卡在鞍点
- **Momentum**：积累速度，穿越局部最优
- **RMSprop**：自适应调整不同方向的步长
- **Adam/AdamW**：结合Momentum和RMSprop，最常用

**学习率调度**：
- 训练过程中动态调整学习率
- Warmup + Cosine Decay是流行组合
- Linear Scaling Rule用于调整batch size

### 7.2 实践建议

**首选配置**：
1. 优化器：Adam或AdamW
2. 初始学习率：1e-3或5e-4
3. 调度策略：Warmup + Cosine Decay
4. Beta值：$\beta_1=0.9, \beta_2=0.999$

**调试技巧**：
- 绘制损失曲线判断学习率是否合适
- 使用梯度检查验证反向传播实现
- 从小模型开始验证流程

### 7.3 二阶优化方法（简介）

**牛顿法（Newton's Method）**：
- 使用Hessian矩阵（二阶导数）
- 拟合局部二次函数
- 更精确的步长选择

**局限性**：
- Hessian矩阵计算和存储开销巨大（$O(D^2)$）
- 对于大规模深度学习模型不实用
- 仅在小模型上可行

![01:04:55](../../assets/lecture-3/screenshot-01KG6AVXTF2XSWX9WCB3ASZEM4.png)
[01:04:55](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:05:28](../../assets/lecture-3/screenshot-01KG6AWVD2QY2XS360MBZ9KBD4.png)
[01:05:28](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:05:59](../../assets/lecture-3/screenshot-01KG6AXPEWW79K4QPB6WXKK2MT.png)
[01:05:59](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

### 7.4 展望：神经网络

**下一讲主题**：多层神经网络

**核心思想**：
- 堆叠多个线性层
- 层间插入非线性激活函数（如ReLU）
- 能够学习非线性决策边界

**示例**：
$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

**优势**：
- 可以处理线性不可分数据
- 通过多层变换学习复杂特征表示

---

![01:08:02](../../assets/lecture-3/screenshot-01KG6B13VAK1R3T4HGE3T00AJG.png)
[01:08:02](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)

![01:08:31](../../assets/lecture-3/screenshot-01KG6B1X5PKSHGBRCGF59FXTN8.png)
[01:08:31](https://www.bilibili.com/video/BV1YJ3PzLEiW/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2c95328d728d9b9a500883028f75d2f3)



## 参考资料

1. **CS231n课程笔记**：http://cs231n.github.io/
2. **优化算法论文**：
   - Adam: Kingma & Ba, 2014
   - AdamW: Loshchilov & Hutter, 2017
3. **相关概念**：
   - 条件数（Condition Number）
   - Hessian矩阵
   - 凸优化（Convex Optimization）

---

## 附录：重要公式汇总

**损失函数**：
$$L = \frac{1}{N} \sum_{i=1}^{N} L_i + \lambda R(\mathbf{W})$$

**L2正则化**：
$$R(\mathbf{W}) = \sum_{i,j} W_{i,j}^2$$

**L1正则化**：
$$R(\mathbf{W}) = \sum_{i,j} |W_{i,j}|$$

**梯度下降**：
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \nabla_{\mathbf{W}} L$$

**Momentum**：
$$\mathbf{v}_{t+1} = \rho \mathbf{v}_t + \nabla_{\mathbf{W}} L, \quad \mathbf{W}_{t+1} = \mathbf{W}_t - \alpha \mathbf{v}_{t+1}$$

**Adam**：
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla L$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla L)^2$$
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

