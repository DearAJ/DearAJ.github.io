---
date: 2025-05-14T11:00:59-04:00
description: "该论文提出了GoG，用于在不完整知识图谱（IKGQA）上进行问答任务。GoG将LLM同时作为代理和知识图谱，通过“思考-搜索-生成”框架动态整合外部知识图谱和LLM的内部知识。该方法有效缓解了LLM的知识不足和幻觉问题，为复杂问答任务提供了新思路。"
featured_image: "/images/paper-GoG/tom.jpg"
tags: ["paper"]
title: "「论文阅读」Generate-on-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering"
---

[这篇论文](https://arxiv.org/abs/2404.14741)提出了一种称为 Generate-on-Graph(GoG) 的免训练方法，它可以在探索 KG 时，生成新的事实三元组。

具体来说，在不完全知识图谱(IKGQA) 中，GoG 通过 Thinking-Searching-Generating 框架进行推理，它将 LLM 同时视为 Agent 和 KG。

### 1 Introduction

![1](/images/paper-GoG/1.png)

<!--more-->

+ #### KG + LLM 结合的方法：

  + **语义解析(SP)方法**：使用 LLMs 将 nlp 问题转换为逻辑查询，然后通过在 KG 上执行这些逻辑查询来获得答案。

    ![2](/images/paper-GoG/2.png)

  + **检索增强(RA)方法**：从 KG 检索与问题相关的信息，作为外部知识以指导 LLMs 生成答案。

    ![3](/images/paper-GoG/3.png)

  + **Generate-on-Graph**：Thinking-Searching-Generating

    1. **思考**：LLMs分解问题，并确定是否进行进一步搜索 或 根据当前状态生成相关三元组
    2. **搜索**：LLMs 使用预定义的工具（如 a KG engineer executing SPARQL queries）探索 KG 并过滤掉不相关的三元组。
    3. **生成**：LLMs 根据探索的子图，利用其内部知识和推理能力生成所需的新事实三元组并进行验证。

    GoG 重复上述步骤，直到获得足够的信息来回答问题。

    ![4](/images/paper-GoG/4.png)

&nbsp;

### 2 Related Work

+ #### Question Answering under Incomplete KG

  通过相似性分数训练 KG 嵌入层来在 incomplete KG下预测答案。

+ #### Unifying KGs and LLMs for KGQA

  1. **语义解析 （SP） 方法**

     使用 LLMs将问题转换为结构查询。然后，KG 引擎可以执行这些查询，以根据 KG 得出答案。

     缺点：有效性在很大程度上取决于生成的查询的质量和 KG 的完整性。

  2. **检索增强 （RA） 方法**

     从 KG 中检索相关信息以提高推理性能。

+ #### LLM reasoning with Prompting

  DecomP：通过将复杂任务分解为更简单的子任务，并将它们委托给特定于LLMs子任务来解决。

  ReAct：LLMs 将 ReAct 视为与环境交互、并决定从外部来源检索信息的代理。

&nbsp;

### 3 Generate-on-Graph (GoG) 

![5](/images/paper-GoG/5.png)

1. #### Thinking

   **将 LLM 作为与环境交互的代理以解决任务。**

   + Thought 1：对于每个步骤 i ，GoG 首先生成一个思想 ti∈ℒ （ℒ 是语言空间）以分解原始问题，

   + Thought 2：并决定哪一个子问题应该下一个被解决
   + Thought 4：或确定它是否有足够的信息来输出最终答案。
   + Action 1, 2：然后，基于这个想法 ti ，GoG 生成一个动作 ai∈𝒜 （𝒜 是动作空间）从 KG 中搜索信息
   + Action 3：或通过推理和内部知识生成更多信息

   &nbsp;

2. #### Searching

   根据最终的想法 ti ，从目标实体 ei 的相邻实体中找到最相关的 top-k 实体 Ei。

   + **Exploring**：GoG 首先使用预定义的 SPARQL queries 来获取链接到与目标实体 ei 连接的所有关系 Ri。
   + **Filtering**：检索关系集 Ri 后，根据最后的想法 ti ，LLMs 被用于选择最相关的前 N 关系 Ri′ 。

   最后，根据目标实体 et 和相关关系集 Ri′ 获取最相关的实体集 Ei 

   &nbsp;

3. #### Generating

   + **Choosing**：使用 BM25 Robertson 和 Zaragoza 从以前的观测中检索最相关的三元组。
   + **Generating**：检索到相关三元组后，LLMs用于根据这些相关三元组及其内部知识生成新的事实三元组。生成过程将重复 n 多次，以尽量减少错误和幻觉。
   + **Verifying**：用 LLMs 来验证生成的三元组，并选择那些更有可能准确的作为 Observation。

   还可以LLMs生成以前未探索过的实体，将实体链接到 KG 中相应的机器标识符 （MID）。

重复上述三个步骤，直到获得足够的信息，然后以 F⁢i⁢n⁢i⁢s⁢h⁢[ea] 的形式输出最终的答案（ea 代表答案实体）。

&nbsp;

### 主要贡献

1. 提出了利用LLMs在不完整的知识图谱中进行问答的方法，并构建了相应的基准数据集。
2. 提出了 Generate-on-Graph （GoG），它使用 Thinking-Searching-Generating 框架来解决 IKGQA。
3. 两个数据集上的实验结果表明了 GoG 的优越性，并证明 LLMs 可以与 IKGQA 相结合来回答复杂的问题。
