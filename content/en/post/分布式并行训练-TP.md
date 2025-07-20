---
date: 2026-07-14T11:00:59-04:00
description: "https://docs.pytorch.org/tutorials/distributed/home.html"
featured_image: "/images/DPT-TP/jaz.png"
tags: ["pytorch", "AI Infra"]
title: "分布式并行训练 - TP"
---

## Tensor Parallel

**张量并行** (TP) 是一种用于训练大规模 Transformer 模型的高效并行技术，[序列并行](https://arxiv.org/abs/2205.05198) (SP) 是 TP 并行的一种变体，它在序列维度上对 `nn.LayerNorm` 或 `RMSNorm` 进行分片，以在训练期间进一步节省激活内存。随着模型变大，激活内存成为瓶颈，因此在张量并行训练中，通常将序列并行应用于 `LayerNorm` 或 `RMSNorm` 层。

![1](/Users/aijunyang/DearAJ.github.io/static/images/DPT-TP/1.png)



### TP 的工作原理

+ **分片初始化**

  - 确定将哪种 `ParallelStyle` 应用于每一层，并通过调用 `parallelize_module` 对初始化后的模块进行分片

  - 并行化后的模块将把其模型参数替换为 DTensors，DTensor 将负责使用分片计算运行并行化后的模块

+ **运行时前向/反向传播**

  - 根据用户为每种 `ParallelStyle` 指定的输入/输出 DTensor 布局，它将运行适当的通信操作来转换输入/输出的 DTensor 布局（例如 `allreduce`、`allgather` 和 `reduce_scatter`）

  - 为并行化的层运行分片计算，以节省计算/内存（例如，`nn.Linear`、`nn.Embedding`）

### 何时以及为何应该应用张量并行





