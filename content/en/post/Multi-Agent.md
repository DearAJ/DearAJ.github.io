---
date: 2026-08-12T10:00:59-04:00
description: ""
featured_image: "/images/multimodal/jaz.png"
tags: ["multimodal", "LLM"]
title: "Multi Agent"
---

## MDAgents（NeurIPS 2024 Oral）









## MedAgents（ACL 2024 Findings）







## Enhancing diagnostic capability with multi-agents conversational large language models（npj25）



## MMedAgent-RL





*# 使用RAG链处理问题*

​            result = rag_chain.invoke({"input": question, "chat_history": []})

​            

​            *# 获取检索到的上下文*

​            retrieved_contexts = []

​            *if* "context" in result:

​                *for* doc *in* result["context"]:

​                    retrieved_contexts.append(doc.page_content)

​            

​            *# 构建结果记录*

​            result_record = {

​                "user_input": question,

​                "response": result["answer"],

​                "retrieved_contexts": expected_answer  *# 使用qa_pairs中的answer作为retrieved_contexts*

​            }