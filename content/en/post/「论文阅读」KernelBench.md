---
date: 2026-07-02T11:00:59-04:00
description: ""
featured_image: "/images/paper-KernelBench/jaz.png"
tags: ["paper"]
title: "「论文阅读」KernelBench: Can LLMs Write Efficient GPU Kernels"
---

**目标**：**评估** LLMs 在生成**高性能 GPU 内核代码**（特别是 CUDA 代码）上的能力。

**KernelBench**：首个专门**评估** LLM 生成高性能 GPU 内核能力的基准测试。

**新的评估指标 fastp**：衡量生成的正确内核的百分比，这些内核提供的加速比大于基线的可调整阈值 p 。

&nbsp;

KernelBench 在三个级别的 AI 工作负载上测试 LM 优化：

1.  **Individual operations: **包括各种 AI 运算符、包括矩阵乘法、卷积、激活、范数和损失。
2. **Sequence of operations:** provide problems that contain 3-6 individual operations together
3. **端到端架构**：architectures from popular AI repositories 

&nbsp;

<!--more-->

## 研究背景

1. **研究问题:** 这篇文章要解决的问题是如何利用语言模型（LMs）自动生成高效的GPU内核。当前的GPU内核编写既耗时又需要专业知识，而机器学习架构的快速发展使得这一问题更加紧迫。
2. **研究难点:** 该问题的研究难点包括：生成功能正确且性能优越的内核、处理不同硬件平台的特定指令集和优化技术、以及在大规模数据上进行有效的训练。
3. **相关工作:** 相关工作包括现有的内核编程库（如cuDNN、CUTLASS、Apple MLX）、编译器工具（如torch.compile、FlexAttention）以及使用语言模型进行代码生成的尝试。

## 研究方法

这篇论文提出了一个名为KernelBench的框架，用于**评估语言模型在生成高效GPU内核方面的能力**。具体来说，KernelBench包含以下内容：

1. **任务格式:** KernelBench包含250个任务，涵盖单个操作、操作序列和端到端架构三个层次。每个任务提供一个PyTorch参考实现，并要求语言模型生成优化后的CUDA内核。

2. 任务选择:

    任务分为三个级别：

   - **Level 1:** 单个原始操作，如矩阵乘法、卷积、激活函数等。
   - **Level 2:** 操作序列，如卷积后接 ReLU 和偏置。
   - **Level 3:** 完整的机器学习架构，如 AlexNet 和 MiniGPT。

3. **评估指标:** 提出了一个新的评估指标fastp，定义为生成的内核在功能正确且速度提升超过阈值p的任务比例。公式如下：

   $$   \text{fast}_{p}=\frac{1}{N}\sum_{i=1}^{N} 1\left(\text{correct}_{i}\wedge\{\text{speedup}_{i}>p\}\right)   $$

   其中，fast$_{0}$表示内核功能正确的比例。

## 实验设计

实验设计包括以下几个方面：

1. **数据收集:** 使用250个精心挑选的PyTorch机器学习工作负载作为基准测试任务。
2. **实验设置:** 在NVIDIA L40S GPU上进行实验，使用Python 3.10、PyTorch 2.5.0+cu124和CUDA 12.4环境。
3. **样本选择:** 对每个任务，提供PyTorch参考实现，并要求语言模型生成优化后的CUDA内核。
4. **参数配置:** 使用不同的温度参数进行采样，如DeepSeek-V3使用温度1.6，Llama 3.1-70B使用温度0.7。

## 结果与分析

实验结果表明：

1. **整体表现:** 当前最先进的语言模型在KernelBench上的表现不佳，匹配PyTorch基线的任务不到20%。
2. **错误分析:** 大部分模型生成的核存在执行错误和功能不正确的问题。推理模型（如OpenAI-o1和DeepSeek-R1）产生的执行失败较少，但在功能正确性上与其他模型相似。
3. **性能分布:** 少数模型生成的核在某些任务上表现出显著的速度提升，但整体上，大多数生成的核速度较慢。
4. **硬件通用性:** 在不同GPU类型上，模型生成的核表现差异较大，特别是在Level 2任务上。

## 总体结论

这篇论文通过引入KernelBench框架，展示了当前语言模型在生成高效GPU内核方面的挑战和潜力。尽管现有模型在这一任务上表现不佳，但研究表明，通过迭代细化和提供硬件信息，可以显著提高生成内核的质量和性能。未来的工作可以集中在改进模型的推理能力和探索更高效的编程抽象。