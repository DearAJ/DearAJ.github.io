---
date: 2025-07-02T11:00:59-04:00
description: " AlphaEvolve 是一种进化编码代理，可显着增强 LLM 在挑战性任务上的能力（如解决开放的科学问题、优化计算基础设施）"
featured_image: "/images/paper-AlphaEvolve/jaz.png"
tags: ["paper"]
title: "「论文阅读」AlphaEvolve: A coding agent for scientific and algorithmic discovery"
---

[*AlphaEvolve*](http://arxiv.org/abs/2506.13131) 使用进化方法，不断接收来自一个或多个评估者的反馈，迭代改进算法，从而有可能带来新的科学和实践发现。

## Introduction

AlphaEvolve represents the candidates (for example, new mathematical objects or practical heuristics) as algorithms and **uses a set of LLMs to generate, critique, and evolve** a pool of such algorithms.

![1](/images/paper-AlphaEvolve/1.png)

## AlphaEvolve

![2](/images/paper-AlphaEvolve/2.png)

<!--more-->

1. 用户提供初始程序（带有待发展的组件标记）、评估代码和可选配置
2. AlphaEvolve 启动进化循环
   + ***Prompt sampler*** 使用 Program 数据库中的程序来构建丰富的提示
   + 根据这些提示，***LLMs ensemble*** 生成代码修改 （diffs），用于创建新程序
   + ***Evaluators*** 对这些项目进行评分
   + 并将好的解决方案重新注册回 ***Program database*** 中，推动迭代发现越来越好的程序

![3](/images/paper-AlphaEvolve/3.png)

&nbsp;

### 1 Task specification

#### Evaluation

用户需提供自动评估生成解决方案的机制（函数h），其形式为将解决方案映射到一组标量评估指标的函数，通常以Python函数实现。*AlphaEvolve* 基于此解决问题。

#### API

支持用户通过输入API对代码库中的代码块进行注释，以便集成到现有代码库中，作为 AlphaEvolve 的初始解决方案。

![4](/images/paper-AlphaEvolve/4.png)

1. a. 用户提供的文件，其中包含标记为 evolution 的块，以及可以调用以对当前版本代码进行评分的特殊 evaluate 函数

   b. Example of an assembled prompt to be provided to the LLMs

   c. LLM 生成的示例输出.

2. （c） 中建议的 diffs 将应用于提示 （b） 中显示的“当前程序”，修改后的程序将发送给评估人员

   评估者将从 （a） 中调用 evaluate 函数，以获得新提议的程序的分数

#### Flexibility in choosing the abstraction

AlphaEvolve 可以以同的方式解决同一个问题

### 2 Prompt sampling

利用SOTA LLMs，支持多种类型的自定义，并提供长上下文作为主要进化提示的一部分。

提示包括：多个从程序数据库采样的先前发现的解决方案、系统指令、随机格式化、渲染的评估结果以及元提示进化等内容。

### 3 Creative generation

利用 SOTA LLMs 的能力，消化有关先前开发解决方案的信息并提出新的多样化改进方案。

### 4 Evaluation

每个新提出的解决方案都会自动评估——相当于简单地对生成的解**执行用户提供的评估函数 h** 

- 支持可选机制以提高评估的灵活性和效率：
  - 评估级联（假设检验）：指定难度递增的测试用例集成，只有当新解决方案在所有早期阶段都取得了足够有希望的结果时，才会在下一阶段进行评估
  - LLM-生成的反馈：使用单独的LLM调用进行评分，并添加到分数字典中以指导进化，或者当不满足标准时，它们可用于丢弃解决方案
  - 平行化评估：允许 AlphaEvolve 通过对评估集群的异步调用来分发工作

+ 允许优化多个用户提供的分数，即在一个或多个评估指标下获得高分的进化对象

### 5 Evolution

进化过程中不断生成带有评估结果的解决方案，并存储在进化数据库中，旨在优化重新利用先前探索的想法，同时保持多样性以鼓励探索整个搜索空间

数据库实现的算法受 MAP elites 算法和基于岛屿的种群模型的启发

### 6 分布式

AlphaEvolve 被设计为**异步计算管道**，由 *控制器、LLM采样器和评估节点* 组成，优化整体吞吐量而非单个计算的速率，以在特定计算预算内最大化提出和评估的想法数量。

&nbsp;

&nbsp;

## Results

#### 加速矩阵乘法

- 从基本梯度算法出发，AlphaEvolve能开发出超越现有方法的复杂张量分解算法。
- 在表2中展示了针对14个不同矩阵乘法目标的改进结果，特别是在4×4复值矩阵乘法上取得了突破，发现了使用48次标量乘法的算法，这是56年来首次在该设置上超过Strassen算法。

#### 解决数学问题

- 应用于50多个数学问题，涵盖分析、组合学、数论和几何等多个分支，在75%的情况下重新发现了最佳已知构造，在20%的情况下发现了比之前已知最佳构造更好的新对象，从而改进了SOTA。
- 例如在分析方面改进了自相关不等式的最佳已知界限，在组合学和数论方面对Erdős的最小重叠问题建立了新的上界，在几何和打包方面改善了11维中的吻数问题等。

#### 优化谷歌计算生态系统

- **数据中心调度**：将在线作业调度问题视为向量装箱问题，AlphaEvolve发现了一个简单而有效的启发式函数，应用于谷歌数据中心的模拟器后，在测试数据集上表现优于生产中的启发式函数，部署后可平均恢复谷歌车队0.7%的计算资源。
- **Gemini内核工程**：优化用于训练Gemini的重要矩阵乘法内核的平铺策略，AlphaEvolve迭代探索和细化平铺启发式，使所有内核的平均运行速度提高了23%，Gemini的整体训练时间减少了1%，并将内核优化时间从数月缩短至数天。
- **硬件电路设计**：优化TPU算术电路，AlphaEvolve找到了一个简单的代码重写，去除了不必要的位，该改进已集成到即将推出的TPU中，代表了Gemini对TPU算术电路的首次直接贡献。
- **编译器生成代码优化**：直接优化XLA生成的IRs封装的FlashAttention内核及预处理和后处理代码，使感兴趣配置的FlashAttention内核速度提高了32%，预处理和后处理部分速度提高了15%。

&nbsp;

### 5 Ablations 消融实验

对两个任务（寻找更快矩阵乘法的张量分解和计算吻数的下界）进行了消融实验，以了解AlphaEvolve各组成部分的有效性。

- 结果表明，进化方法、提示中的上下文、元提示、全文件进化以及强大的语言模型等组件都对结果有显著改进。
