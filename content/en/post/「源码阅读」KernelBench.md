---
date: 2025-07-20T11:00:59-04:00
description: "https://github.com/ScalingIntelligence/KernelBench"
featured_image: "/images/code-KernelBench/jaz.png"
tags: ["paper"]
title: "「源码阅读」KernelBench"
---

## 任务描述

构建 KernelBench 有 4 个级别的任务：

- **Level 1 🧱**: 单核算子(100 个问题)，如卷积、矩阵乘法、层归一化
- **Level 2 🔗**: 简单融合模式(100 个问题)，如 Conv + Bias + ReLU，Matmul + Scale + Sigmoid
- **Level 3 ⚛️**: 端到端全模型架构(50个问题)，如MobileNet、VGG、MiniGPT、Mamba）
- **Level 4 🤗**: Hugging Face 优化过的整个模型架构

## 评估方法

+ **正确性检查✅**：确保模型生成的 kernel 在功能上与参考实现（如 PyTorch 的官方算子）完全一致。进行 `n_correctness` 次测试。

+  **性能评估⏱️**：验证生成的 kernel 是否比参考实现更高效。重复 `n_trial` 次消除偶然误差。指标是加速比。

实现代码位于： `src/eval.py` 

**评估脚本： `scripts/run_and_check.py`** 

### 总基准指标

1. **`fast_p`** ：既正确又加速大于阈值的任务的分数 `p` 。提高加速阈值 `p` 可使任务更具挑战性。

2. **加速比**：PyTorch 参考实现运行时间 与 生成的内核时间 之比。

**计算整体基准测试性能脚本： `scripts/greedy_analysis.py`** 

<!--more-->&nbsp;

## 代码结构

```
KernelBench/
├── assets/
├── KernelBench/ # Benchmark dataset files
├── src/ 				 # KernelBench logic code
│   ├── unit_tests/  
│   ├── prompts/
│   ├── ....
├── scripts/ 		 # helpful scripts to run the benchmark
├── results/ 		 # baseline times across hardware 
├── runs/ 			 # where your runs will be stored
```

&nbsp;

### `src/`

项目的核心代码。

#### (1) `src/unit_tests/`

- **作用**：单元测试是理解代码功能的“活文档”。
- **重点关注**：
  - 测试覆盖的模块（如正确性检查、性能测试）。
  - 测试用例的输入输出（如随机输入生成、与参考算子对比的逻辑）。
- **示例**：若存在 `test_correctness.py`，可能包含你提到的“`n_correctness` 次随机测试”的实现。

#### (2) `src/prompts/`

- **作用**：如果项目涉及LLM生成kernel（如通过自然语言描述生成代码），这里可能存放提示词模板。
- **重点关注**：
  - Prompt 的结构（如是否要求模型生成特定类型的 kernel）。
  - 是否有针对不同算子（如矩阵乘法、卷积）的差异化提示。

#### **(3) 其他关键模块**

- **入口文件**：如 `src/main.py` 或 `src/benchmark.py`，了解主流程。
- **核心逻辑**：查找以下功能的实现：
  - Kernel 生成（如调用LLM或优化算法）。
  - 正确性验证（对比PyTorch参考算子）。
  - 性能测试（计时、计算加速比）。

&nbsp;

### **`KernelBench/`**

- 可能是预定义的 benchmark 数据集（如常见算子的输入输出对）。
- 查看文件格式（如JSON/YAML），了解评估标准的数据结构。

### **`assets/`**

- 可能存放静态资源（如图片、配置文件）。
- 检查是否有默认配置（如 `config.yaml`）或硬件规格说明。

&nbsp;

### `scripts/`

包含一键运行测试、生成结果或部署的脚本。

- 脚本的输入参数（如 `run_benchmark.py --kernel=matmul`）。
- 是否封装了正确性检查（`n_correctness`）和性能测试（`n_trial`）的参数。

#### generate_and_eval_single_sample.py

**核心功能**：针对指定的 GPU 计算问题（如矩阵乘法），生成优化的内核代码（CUDA 或 Triton），并评估其正确性和性能。

- **流程**：
  1. **加载问题**：从本地或 Hugging Face 数据集获取问题描述和参考代码。
  2. **生成代码**：通过 LLM 根据问题描述生成自定义内核代码。
  3. **评估代码**：对比生成的代码与参考代码，验证功能正确性并测量性能（如加速比）。





&nbsp;

###  结果 `results/`

- 预计算的基线数据（如PyTorch官方算子在多种硬件上的性能）。
- 帮助理解“性能达标”的具体阈值（如加速比>1.2x）。

### 基准 `runs/`

- 用户运行生成的日志和结果。
- 查看最新运行的目录，观察输出格式（如时间、正确性统计）。

&nbsp;

&nbsp;

&nbsp;

# [KernelBook 数据集](https://huggingface.co/datasets/GPUMODE/KernelBook)

`dataset_permissive{.json/.parquet}` 是一对 PyTorch 程序和等效的 Triton 代码（由 torch Inductor 生成），可用于训练模型将 PyTorch 代码转换为 Triton 代码。