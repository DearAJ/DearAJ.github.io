---
date: 2026-06-28T10:00:59-04:00
description: ""
featured_image: "/images/mcp/jaz.png"
tags: ["agent", "LLM"]
title: "MCP"
---

## 概述

+ MCP 通过提供标准接口，将 M×N 问题转化为 M+N 问题：

  ![1](/Users/aijunyang/DearAJ.github.io/static/images/mcp/1.png)

  ![2](/Users/aijunyang/DearAJ.github.io/static/images/mcp/2.png)

  每个 AI 应用程序实现 MCP 的客户端一次，每个工具/数据源实现服务器端一次。这大大降低了集成复杂性和维护负担。

### 框架组件

![3](/Users/aijunyang/DearAJ.github.io/static/images/mcp/3.png)

Host：面向用户的 AI 应用程序，最终用户可以直接与之交互。

Client：管理与特定 MCP 服务器的通信。 

Server：一个外部程序或服务，通过 MCP 协议向 AI 模型公开功能。

#### 通信流程
