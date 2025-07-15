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
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        多头注意力机制实现
        
        参数:
            d_model: 输入向量的维度
            num_heads: 注意力头的数量
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model    # 模型总维度（如512）
        self.num_heads = num_heads # 注意力头数量（如8）
        self.d_k = d_model // num_heads  # 每个头的维度（如512/8=64）
        
        # 查询(Query)、键(Key)、值(Value)的线性变换
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        
        # 输出线性层（合并多头结果）
        self.W_o = nn.Linear(d_model, d_model)
        
        # 防止过拟合的dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子（用于稳定梯度）
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
```

```python
def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        参数:
            q: 查询向量 (batch_size, seq_len, d_model)
            k: 键向量 (batch_size, seq_len, d_model)
            v: 值向量 (batch_size, seq_len, d_model)
            mask: 掩码 (batch_size, seq_len, seq_len)
        
        返回:
            output: 注意力输出 (batch_size, seq_len, d_model)
            attention: 注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = q.size(0)
        
        # 线性变换并分割为多头
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(q.device)
        
        # 应用掩码(如果有)
        if mask is not None:
            mask = mask.unsqueeze(1)  # 为多头增加维度
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力权重到V上
        output = torch.matmul(attention, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(output)
        
        return output, attention

```

**视图变换(view)**：将线性变换后的张量重塑为多头的形式

**维度转置(transpose)**：将头维度移到第1维，便于并行计算

**保持信息完整**：总维度d_model被分割为num_heads × d_k，信息量不变

### 

`transpose(1, 2)` - 维度转置

**作用**：交换张量的第1和第2维度

`contiguous()` - 确保内存连续

**作用**：返回一个内存连续的新张量，包含与原始张量相同的数据

`view(batch_size, -1, self.d_model)` - **实际合并多头信息**

**作用**：将张量重新塑形而不改变其数据

**转换前形状**：`(32, 10, 8, 64)`
**转换后形状**：`(32, 10, 512)`

```
# -1自动推断
```
