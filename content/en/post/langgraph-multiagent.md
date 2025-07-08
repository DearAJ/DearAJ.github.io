---
date: 2026-07-05T10:00:59-04:00
description: ""
featured_image: "/images/multi-agent/jaz.png"
tags: ["agent", "LLM"]
title: "Multi-Agent"
---

+ **优势**：
  1. **效率提升**：多个智能体可以同时执行不同子任务，显著缩短任务完成时间
  2. **鲁棒性强**：单个智能体失效时，其他智能体可接管任务
  3. **解决复杂问题**：不同智能体可专注于特定领域
  4. **资源优化**：多个简单智能体可能比单个超级智能体更经济
+ **原因**：足够多的 tokens，调用工具的次数更多，增加并行推理能力。

+ **让不同智能体专注于特定领域**：

  1. 为每个智能体分配明确角色

  2. **分层架构**：

     - **顶层**：协调智能体（Manager Agent）负责任务分配和全局决策。
     - **底层**：领域智能体（Worker Agent）执行具体任务（如工厂中的**物流调度Agent**指挥多个**搬运机器人Agent**）。

  3. 为每个智能体加载特定领域的预训练模型

     将全局知识图谱按领域划分（如智慧医疗中，**疾病诊断Agent**访问医学知识库，**药品交互Agent**访问药理数据库）。

  4. **输入数据筛选**：每个智能体仅接收与其领域相关的输入（如自动驾驶中，**交通标志识别Agent**只处理摄像头数据，**雷达数据处理Agent**只处理雷达信号）。

  5. 在强化学习中，为不同智能体设计领域相关的奖励（如游戏AI中，**攻击型Agent**奖励伤害输出，**治疗型Agent**奖励队友存活率）。

&nbsp;

### Agent Collaboration

![1](/Users/aijunyang/DearAJ.github.io/static/images/multi-agent/1.png)

### Agent Supervisor

![2](/Users/aijunyang/DearAJ.github.io/static/images/multi-agent/2.png)



### Hierarchical Agent Teams

![3](/Users/aijunyang/DearAJ.github.io/static/images/multi-agent/3.png)