---
date: 2025-06-25T10:00:59-04:00
description: ""
featured_image: "/images/difyAgent/jaz.png"
tags: ["agent", "LLM"]
title: "dify - Agent"
---

### 基础实现

![1](/images/difyAgent/1.png)

举例：*WikiAgent*

+ prompt

  ```python
  ***xmi
  ‹instruction>
  - The Al Agent should be knowledgeable about the TV show "The Office".
  - If the question asked is not related to "The Office" or if the Al does not know the answer, it should search for the answer using the Google search tool.
  - The output should not contain any XML tags.
  
  <example>
  - If asked "Who is the regional manager in 'The Office'?", the Al should provide the correct answer.
  - If asked "What year did 'The Office' first premiere?", the Al should provide the correct answer or search for it if unknown.
  ```



### Agent Workflow

#### Prompt Chaining

将任务分解为关键步骤，用gate来验证前面的输出是否符合后续处理的条件。

![2](/images/difyAgent/2.png)

#### Routing

![3](/images/difyAgent/3.png)

#### Parallelization

![4](/images/difyAgent/4.png)



#### Orchestrator-workers

![5](/images/difyAgent/5.png)



#### Evaluator-Optimizer

![6](/images/difyAgent/6.png)

![7](/images/difyAgent/7.png)

![8](/images/difyAgent/8.png)

 