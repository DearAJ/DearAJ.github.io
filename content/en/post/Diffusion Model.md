---
date: 2025-01-17T10:00:59-04:00
description: "扩散模型是一种生成模型，主要用于生成与给定数据集相似的数据。它的核心思想是通过逐步添加噪声（扩散过程）和逐步去噪（逆扩散过程）来学习数据的分布，从而生成高质量的数据样本。"
featured_image: "/images/DM/taytay1.HEIC"
tags: ["Generative AI"]
title: "Diffusion Model"
---

# 概述

## 影像生成模型本质上的共同目标

![19](/images/DM/19.png)

进一步：输入加入了文字表述

目标：产生的图片与真实图片越接近越好

<!--more-->

&nbsp;

## 原理

+ **Reverse Process**（多次Denoise）

  **reconstructing meaningful data from noise** by iteratively removing noise that was added during the **forward process**.

+ **Forward Process**:

  Gradually adds noise to data over multiple steps until the data becomes pure noise.

![1](/images/DM/1.png)

+ Denoise输入：图片 + 噪音程度

  ![2](/images/DM/2.png)

### Denoise Model内部

![3](/images/DM/3.png)

**Noise Predicter**: 预测noise长什么样

+ 如何训练Noise Predicter？

  + 人为创造（加杂讯/Forward Process）

    ![4](/images/DM/4.png)

### **Text-to-Image**

Laion拥有5.85B图片，可进行搜索

输入：图片+噪声程度+**文字叙述**

![5](/images/DM/5.png)

### 算法

![6](/images/DM/6.png)

+ **训练**：
  
  ![17](/images/DM/17.png)
  
  1. sample一张clean image
  2. sample出一个数字
  3. sample出一个noise
  4. **clean image 和 noise 做加权和得到一个有杂讯的图**，然后训练noise predictor（输入有杂讯的图 和 数字；输出目标noise）
  
+ **产生图**：
  
  1. sample一个全是noise图
  
  2. 跑T次：
  
     1. sample另一个noise图
  
     2. 生成x_t（见公式）
  
        ![18](/images/DM/18.png)

&nbsp;

# Stable Diffusion

+ **框架组成**：

  +  **Text Encoder**：输入为文字，输出为向量

    *text encoder对结果的影响很大*

    ![9](/images/DM/9.png)

    读的越多效果越好，不同大小的diffusion model影响不大。

  + **Deneration Model**：输入杂讯和向量，输出中间产物（图片压缩结果）

    + sample出noise，加在中间产物上：

      ![14](/images/DM/14.png)

    + 得到中间产物后，再次Denoise：

      ![15](/images/DM/15.png)

  + **Decoder**：将中间产物还原回原来的图片

    Decoder的训练可以不需要labelled data

    + 训练过程：

      + 若中间产物是小图

        ![12](/images/DM/12.png)

        把收集到的所有的图片缩小，小图变大图

      + 若中间产物是Latent Representation

        ![13](/images/DM/13.png)

        训练一个Auto-encoder，使输入与输出越接近越好。最后，再将Decoder拿出来用即可。

  通常，3个Model分开训练，最后组合在一起

![7](/images/DM/7.png)

+ **经典结构**：

  1. Stable Diffusion

     ![8](/images/DM/8.png)

     + encoder：处理输入文字等
     + diffusion model
     + decoder：还原

  2. DALL-E系列：

     + encoder
     + autogressive model 和 diffusion model
     + decoder

  3. Imagen：

     + encoder
     + diffudion model：生成人类可看懂的中间结果
     + decoder

### v.s VAE

![16](/images/DM/16.png)

&nbsp;

&nbsp;

# **评估指标**

+ **KL 散度**

  度量两个概率分布之间的差异程度

+ **Fréchet Inception Distance（FID）**

  评估影响生成模型的好坏。

  ![10](/images/DM/10.png)

  + 现有一个pre-train好的分类CNN Model，然后将图片扔入网络，得到生成图片。计算两组生成和真实图片之间的Fréchet distance。距离越小越好。

  + 缺点：想要生成大量images。

+ **Contrastive Language-Image Pre-Training**

  **(CLIP)**

  是用400 million image-text pairs训练出来的模型。

  ![11](/images/DM/11.png)

  + 计算产生的图片、输入的文字丢进CLIP，计算CLIP输出向量之间的距离。

![20](/images/DM/20.png)





















