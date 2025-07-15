---
date: 2025-07-09T11:00:59-04:00
description:  ""
featured_image: "/images/ml/jaz.png"
tags: ["ML"]
title: "machine learning"
---

好久没碰又忘了...😣 复习复习！！！

## logistic 逻辑回归模型

### 1. sigmoid function (logistic function)

![1](/images/ml/1.png)

输出$f_{w,b}(x) = P(y=1|x;w,b)$：y 等于 1 的概率

> [!NOTE]
>
> 如何理解？
>
> ​	When is $f_{w,b}(x) ≥ 0.5$
>
> ​	g(z) ≥ 0.5
>
> ​	z ≥ 0
>
> ​	w*x + b ≥ 0
>
> **决策边界:** z = w*x+b = 0

<!--more-->

&nbsp;

### 2 逻辑回归的损失函数

平方损失函数会造成函数不凸，下降无法收敛到最小值，不适用。

+ #### 损失函数

  ![2](/images/ml/2.png)

+ #### 代价函数

  ![3](/images/ml/3.png)

+ #### 简化版

  ![5](/images/ml/5.png)

+ #### 梯度下降

  目标：找到合适的 w, b

  ![4](/images/ml/4.png)

  改进：**放缩**（除以最大值 / 均值归一化 / z分数归一化）

&nbsp;

&nbsp;

## softmax 多分类

以**四个**可能的输出值为例：

![9](/images/ml/9.png)

$a_{1} + a_{2} + a_{3} + a_{4} = 1$

 Softmax regression：N 个可能的输出值

 $z_{j} = w_{j} x + b_{j} j = 1, ..., N$
$a_{j} = \frac{e^{Z_{j}}}{\sum_{k=1}^Ne^{Z_{k}}}$
$a_{1} + a_{2} + ... + a_{N} = 1$

+ **损失函数**

  ![10](/images/ml/10.png)

&nbsp;

&nbsp;

## 线性回归

+ #### 损失函数：平方误差

  ![6](/images/ml/6.png)

+ #### 梯度函数

  参数初始值并不重要，通常将它们设置为 0 ：不同方向下坡、找局部最小值

  ![7](/images/ml/7.png)

  ![8](/images/ml/8.png)

  

