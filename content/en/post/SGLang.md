---
date: 2025-07-19T11:00:59-04:00
description: "https://docs.sglang.com.cn"
featured_image: "/images/sglang/jaz.png"
tags: ["inference"]
title: "SGLang"
---

优势：**后端运行快速**、**前端语言灵活**、**模型支持广泛**、**社区活跃**。

SGLang 的核心是通过其 Python API 构建和执行 Language Model Program。

#### **启动服务器**

```python
import sglang as sgl

# 配置 SGLang 运行时
# 如果在本地运行服务，这里指定服务地址和端口
# 如果直接在 Python 进程中加载模型 (需要安装 sglang[srt])，可以使用 sgl.init("model_path")
# 假设此时服务已在本地 30000 端口启动，并加载了模型：
sgl.init("http://127.0.0.1:30000")

# 或者，如果在 Python 进程中直接加载模型（需要足够的显存）
# sgl.init("meta-llama/Llama-3.1-8B-Instruct") # 使用 Hugging Face ID
# 或者
# sgl.init("/path/to/your/model_dir") # 使用本地模型路径
```

1. #### 定义和运行一个简单的生成任务

   `sgl.Runtime()` 是 LM 程序的入口。

   ```python
   import sglang as sgl		# 假设已经通过 sgl.init(...) 初始化了运行时
   
   # 定义一个 LM Program
   @sgl.function
   def simple_gen(s, query):
       s += f"用户问：{query}\n"
       # 使用 sgl.gen() 进行文本生成
       s += "回答：" + sgl.gen("answer", max_tokens=64)	# 使用 sgl.gen() 进行文本生成
   ```

   <!--more-->

   ```python
   state = sgl.Runtime().new_session()						# 创建会话实例 Session 存储上下文和执行程序
   state.run(simple_gen, query="什么是人工智能？")	 # 执行 LM Program
   
   print(state["answer"])												# 获取生成的文本
   ```

   + `@sgl.function` 装饰器：定义了一个 SGLang LM 程序。
   +  `s` 参数：代表当前的会话状态 (session state)，可以通过 `s += ...` 向其中添加文本或指令。
   +  `sgl.gen("answer", max_tokens=64)` ：SGLang 的**核心生成函数**。它会指示模型在这里生成文本，并将生成的文本保存在会话状态的 `answer` 变量中。`max_tokens` 控制生成的最大 token 数。
   +  `sgl.Runtime().new_session()`：创建一个新的会话，每个会话有自己的 KV 缓存。
   +  `state.run(...)` ：执行定义的 LM 程序。
   +  `state["answer"]` ：访问生成结果。

   &nbsp;

2. #### 使用 `gen` 函数及其参数

   `sgl.gen` 函数支持多种参数来控制生成行为，如 `temperature`, `top_p`, `stop` 等。

   ```python
   import sglang as sgl
   
   # 假设已经通过 sgl.init(...) 初始化了运行时
   
   @sgl.function
   def controlled_gen(s, topic):
       s += f"请写一篇关于 {topic} 的短介绍，不超过 50 个字。\n"
       s += "介绍：" + sgl.gen("intro",
                              max_tokens=50,        # 最大生成 token 数
                              temperature=0.8,      # 控制随机性
                              stop=["。", "\n"])   # 遇到句号或换行符停止生成
   
   state = sgl.Runtime().new_session()
   state.run(controlled_gen, topic="机器学习")
   
   print(state["intro"])
   ```

   + `temperature`：生成文本的随机性。值越高，结果越多样；值越低，结果越确定。
   + `stop`：一个字符串列表。模型生成过程中遇到列表中的任何字符串时将停止生成。

   &nbsp;

3. #### 使用 `select` 实现多项选择

   `sgl.select` 函数提供多个选项，并让模型从这些选项中选择一个，预测接下来生成某个词的概率分布。

   ```python
   import sglang as sgl
   
   # 假设已经通过 sgl.init(...) 初始化了运行时
   
   @sgl.function
   def choose_option(s, context):
       s += context + "\n请从 A, B, C 三个选项中选择最合适的一个。\n"
       s += "选项：A. 非常好 B. 一般 C. 不好\n"
       s += "选择：" + sgl.select("choice", ["A", "B", "C"]) # 从列表中选择一个词
   
   state = sgl.Runtime().new_session()
   state.run(choose_option, context="今天的用户体验感觉如何？")
   
   print(f"模型选择的选项是: {state['choice']}")
   ```

   + `sgl.select("choice", ["A", "B", "C"])` 会约束模型，只能生成列表 `["A", "B", "C"]` 中的一个词。模型根据前面的上下文计算生成每个词的概率，并返回概率最高的词。

   这对于实现分类、问卷选择等任务非常有用。

   &nbsp;

4. #### 构建一个多轮对话

   SGLang 的会话 (Session) 本身支持维护多轮对话的状态。只需在同一个 `state` 对象上**多次运行 LM 程序或逐步添加内容**即可。

   ```python
   import sglang as sgl
   
   # 假设已经通过 sgl.init(...) 初始化了运行时
   
   @sgl.function
   def chat_turn(s, user_message):
       s += f"用户: {user_message}\n"
       s += "助手: " + sgl.gen("bot_response", max_tokens=128, stop=["用户:", "\n"])
   
   chat_state = sgl.Runtime().new_session()	# 创建一个新的会话用于对话
   
   # 第一轮对话
   chat_state.run(chat_turn, user_message="你好，你能帮我什么？")
   print(f"助手: {chat_state['bot_response']}")
   
   # 第二轮对话，在同一个 chat_state 上继续
   chat_state.run(chat_turn, user_message="谢谢！")
   print(f"助手: {chat_state['bot_response']}")
   
   # 第三轮对话
   chat_state.run(chat_turn, user_message="再见。")
   print(f"助手: {chat_state['bot_response']}")
   
   print("\n--- 完整对话历史 ---")
   print(chat_state.text())			 # 可通过 chat_state.text() 查看完整的对话历史（prompt）
   ```

   + `chat_turn` 函数处理单轮对话：

     1. 接收用户消息，添加 `用户:` 前缀
     2. 然后使用 `sgl.gen` 生成 `助手:` 的回复

   + 每次调用 `chat_state.run()` 都在同一个 `chat_state` 对象上进行。

     SGLang 会自动将之前的对话内容保留在会话的 KV 缓存中，确保模型能够感知完整的对话历史。

   &nbsp;

5. #### JSON 模式强制输出

   SGLang 可通过**指定 BNF 语法**或利用**模型自身的 JSON 模式**能力来强制输出 JSON 格式。

   > **BNF基本结构**：`<符号> ::= 定义`

   如下是一种常见的 *利用提示工程结合生成约束* 实现 JSON 输出的方式：
   
   ```python
   import sglang as sgl
   import json 						# 用于解析输出
   
   # 假设已经通过 sgl.init(...) 初始化了运行时
   
   @sgl.function
   def gen_json_output(s, item_name):
       s += f"请提供关于 '{item_name}' 的详细信息，包括名称(name)、类型(type)和描述(description)，以 JSON 格式输出。\n"
       s += "{\n" 																		# 引导模型输出 JSON 结构
       s += "  " + sgl.gen("json_content", stop="}") # 生成 JSON 内容直到遇到 }
       s += "}"
   
   state = sgl.Runtime().new_session()
   state.run(gen_json_output, item_name="笔记本电脑")
   
   # 尝试解析 JSON 输出
   json_string = "{" + state["json_content"] + "}"
   try:
       data = json.loads(json_string)
       print("成功解析 JSON:")
       print(json.dumps(data, indent=2, ensure_ascii=False))
   except json.JSONDecodeError as e:
       print(f"JSON 解析失败: {e}")
       print("原始输出:")
       print(json_string)
   ```
   
   + 通过在提示中明确要求 JSON 格式，并手动添加 `{` 开头和 `}` 结尾，引导模型生成 JSON。
   + `sgl.gen("json_content", stop="}")` 生成 `{` 和 `}` 之间的内容。
   
   > 可查阅 SGLang 官方文档以获取最准确和高级的结构化输出用法。

<!--more-->