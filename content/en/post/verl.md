---
date: 2026-07-21T11:00:59-04:00
description: "https://verl.readthedocs.io/en/latest/#"
featured_image: "/images/verl/jaz.png"
tags: ["RL"]
title: "verl"
---

#### verl 的优势：

+ 灵活扩展各种 RL 算法。
+ 无缝集成现有LLM基础设施和模块化 API。
+ 灵活的设备映射和并行性。
+ 与 HuggingFace 模型集成

#### 后端选择：

1. **Training**：[FSDP](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html) 或 [Megatron-LM](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html)。

2. **Inference**：建议打开` env var VLLM_USE_V1=1` 以获得最佳性能。可使用 [SGLang 后端](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)。