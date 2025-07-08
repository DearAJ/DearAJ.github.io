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

在三个级别的 AI 工作负载上测试 LM 优化：

1.  **Individual operations: **包括各种 AI 运算符、包括矩阵乘法、卷积、激活、范数和损失。
2. **Sequence of operations:** provide problems that contain 3-6 individual operations together
3. 端到端架构：architectures from popular AI repositories 