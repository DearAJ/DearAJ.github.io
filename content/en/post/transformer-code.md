---
date: 2026-06-28T10:00:59-04:00
description: ""
featured_image: "/images/transformerCode/jaz.png"
tags: ["LLM"]
title: "transformer-code"
---

transformer 架构：基于编码器-解码器来处理序列对；基于纯注意力机制。

 &nbsp;

| 类型                | 核心能力             | 优点                   | 缺点                           | 适用场景                 |
| :------------------ | :------------------- | :--------------------- | :----------------------------- | :----------------------- |
| **Self-Attention**  | 同一序列内部关系建模 | 长距离依赖、并行计算   | 计算复杂度高、可能忽略局部结构 | Encoder/Decoder的基础层  |
| **Multi-Head**      | 多子空间联合建模     | 多视角学习、表达能力强 | 计算量增加、头可能冗余         | 增强Self/Cross-Attention |
| **Cross-Attention** | 跨序列关系建模       | 跨序列对齐、信息融合   | 依赖外部序列质量               | Decoder连接Encoder       |

1. **短文本或高精度需求**：优先用Multi-Head Self-Attention（更多头）。
2. **长序列（如文档）**：考虑稀疏注意力（如Longformer）或分块计算降低复杂度。
3. **跨模态任务**：Cross-Attention是必备模块（如视觉问答中文本查询图像特征）。

&nbsp;

### 多头注意力

输入是query和 key-value，注意力机制首先计算query与每个key的关联性（compatibility），每个关联性作为每个value的权重（weight），各个权重与value的乘积相加得到输出。

- **Q（Query）**：用于“主动查询”其他词的信息。
- **K（Key）**：用于“被查询”，决定与其他词的相关性。
- **V（Value）**：携带实际的信息，用于加权求和生成输出。

![1](/Users/aijunyang/DearAJ.github.io/static/images/transformerCode/1.png)

> [!NOTE]
>
> 例如，厨房里有苹果、青菜、西红柿、玛瑙筷子、朱砂碗，每个物品都有一个key（ 维向量）和value（ 维向量）。现在有一个“红色”的query（ 维向量），注意力机制首先计算“红色”的query与苹果的key、青菜的key、西红柿的key、玛瑙筷子的key、朱砂碗的key的关联性，再计算得到每个物品对应的权重，最终输出 =（苹果的权重x苹果的value + 青菜的权重x青菜的value + 西红柿的权重x西红柿的value + 玛瑙筷子的权重x玛瑙筷子的value + 朱砂碗的权重x朱砂碗的value）。最终输出包含了每个物品的信息，由于苹果、西红柿的权重较大（因为与“红色”关联性更大），因此最终输出受到苹果、西红柿的value的影响更大。

&nbsp;



```python
import math
import torch
from torch import nn
from d21 import torch as d2l
```

选择缩放点积注意力作为每一个注意力头

```python
class MultiHeadAttention(nn.Module):
  def _init_(self, key_size, query_size, value_size,
            num_hiddens, nums_heads, dropout, bias=False, **kwargs):
    super(MultiHeadAttenion, self)._init_(**kwargs)
    self.num_heads = nums_heads
    self.attention = d2l.DotProductAttention(dropout)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
    self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
    self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
  
  def transpose_qkv(x, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])
  
  def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
  
  def forward(self, queries, keys, values, valid_lens):
    queries = transpose_qkv(self.W_q(queries), self.num_heads)
    keys = transpose_qkv(self.W_k(queries), self.num_heads)
    values = transpose_qkv(self.W_v (queries), self.num_heads)
    
    if valid_lens is not None:
      valis_lens = torch.repeat_interleave(valid_lens, 
                                           repeats=self.num_heads, 
                                           dim=0)
    
    output = self.attention(queries, keys, values, valid_lens)
    output_concat = transpose_output(output, self.num_heads)
    
    return self.W_o(output_concat)
```



### 

```

```

