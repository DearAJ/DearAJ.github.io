---
date: 2025-06-12T10:00:59-04:00
description: "Prompt 是用户输入给 AI 模型的指令或问题，用于引导 AI 生成相应的回答或内容。"
featured_image: "/images/prompt/tomo.jpg"
tags: ["LLM"]
title: "Prompting Guide"
---

可以在[langchainhub](https://smith.langchain.com/hub/)上找prompt

## [1. Agentic Workflows](https://cookbook.openai.com/examples/gpt4-1_prompting_guide?featured_on=talkpython#1-agentic-workflows)

### System Prompt Reminders

在提示中包含三种关键类型的提醒：

1. #### 持久性

   确保模型理解它正在进入多消息轮次，并防止它过早地将控制权交还给用户。示例如下：

   ```
   You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
   ```

2. #### 工具调用

   鼓励模型充分利用其工具，并降低其产生幻觉或猜测答案的可能性。示例如下：

   ```
   If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
   ```

3. #### 规划 [可选]

   可确保模型在文本中明确规划和反映每个工具调用，而不是通过将一系列单独的工具调用链接在一起来完成任务。示例如下：

   ```
   You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
   ```

&nbsp;

### Tool Calls

使用 API 解析的工具描述，将 tools 作为参数传递。

应清楚地命名工具以表明其用途，并在工具的 “description” 字段中添加清晰、详细的描述。

&nbsp;

### Prompting-Induced Planning & Chain-of-Thought

使用 Planning 提示组件来诱导模型在工具调用之间进行规划和反思，生成一个明确的、循序渐进的计划。

&nbsp;

### Sample Prompt: SWE-bench Verified

通用模式如下，包含有关工作流程和问题解决策略的详细说明。

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get(
        "OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"
    )
)

SYS_PROMPT_SWEBENCH = """
You will be tasked to fix an issue from an open-source repository.

Your thinking should be thorough and so it's fine if it's very long. You can think step by step before and after each action you decide to take.

You MUST iterate and keep going until the problem is solved.

You already have everything you need to solve this problem in the /testbed folder, even without internet connection. I want you to fully solve this autonomously before coming back to me.

Only terminate your turn when you are sure that the problem is solved. Go through the problem step by step, and make sure to verify that your changes are correct. NEVER end your turn without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.

Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Workflow

## High-Level Problem Solving Strategy

1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. Debug as needed. Use debugging techniques to isolate and resolve issues.
6. Test frequently. Run tests after each change to verify correctness.
7. Iterate until the root cause is fixed and all tests pass.
8. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.

Refer to the detailed sections below for more information on each step.

## 1. Deeply Understand the Problem
Carefully read the issue and think hard about a plan to solve it before coding.

## 2. Codebase Investigation
- Explore relevant files and directories.
- Search for key functions, classes, or variables related to the issue.
- Read and understand relevant code snippets.
- Identify the root cause of the problem.
- Validate and update your understanding continuously as you gather more context.

## 3. Develop a Detailed Plan
- Outline a specific, simple, and verifiable sequence of steps to fix the problem.
- Break down the fix into small, incremental changes.

## 4. Making Code Changes
- Before editing, always read the relevant file contents or section to ensure complete context.
- If a patch is not applied correctly, attempt to reapply it.
- Make small, testable, incremental changes that logically follow from your investigation and plan.

## 5. Debugging
- Make code changes only if you have high confidence they can solve the problem
- When debugging, try to determine the root cause rather than addressing symptoms
- Debug for as long as needed to identify the root cause and identify a fix
- Use print statements, logs, or temporary code to inspect program state, including descriptive statements or error messages to understand what's happening
- To test hypotheses, you can also add test statements or functions
- Revisit your assumptions if unexpected behavior occurs.

## 6. Testing
- Run tests frequently using `!python3 run_tests.py` (or equivalent).
- After each change, verify correctness by running relevant tests.
- If tests fail, analyze failures and revise your patch.
- Write additional tests if needed to capture important behaviors or edge cases.
- Ensure all tests pass before finalizing.

## 7. Final Verification
- Confirm the root cause is fixed.
- Review your solution for logic correctness and robustness.
- Iterate until you are extremely confident the fix is complete and all tests pass.

## 8. Final Reflection and Additional Testing
- Reflect carefully on the original intent of the user and the problem statement.
- Think about potential edge cases or scenarios that may not be covered by existing tests.
- Write additional tests that would need to pass to fully validate the correctness of your solution.
- Run these new tests and ensure they all pass.
- Be aware that there are additional hidden tests that must also pass for the solution to be successful.
- Do not assume the task is complete just because the visible tests pass; continue refining until you are confident the fix is robust and comprehensive.
"""

PYTHON_TOOL_DESCRIPTION = """This function is used to execute Python code or terminal commands in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail. Just as in a Jupyter notebook, you may also execute terminal commands by calling this function with a terminal command, prefaced with an exclamation mark.

In addition, for the purposes of this task, you can call this function with an `apply_patch` command as input.  `apply_patch` effectively allows you to execute a diff/patch against a file, but the format of the diff specification is unique to this task, so pay careful attention to these instructions. To use the `apply_patch` command, you should pass a message of the following structure as "input":

%%bash
apply_patch <<"EOF"
*** Begin Patch
[YOUR_PATCH]
*** End Patch
EOF

Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.

*** [ACTION] File: [path/to/file] -> ACTION can be one of Add, Update, or Delete.
For each snippet of code that needs to be changed, repeat the following:
[context_before] -> See below for further instructions on context.
- [old_code] -> Precede the old code with a minus sign.
+ [new_code] -> Precede the new, replacement code with a plus sign.
[context_after] -> See below for further instructions on context.

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change's [context_after] lines in the second change's [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

Note, then, that we do not use line numbers in this diff format, as the context is enough to uniquely identify code. An example of a message that you might pass as "input" to this function, in order to apply a patch, is shown below.

%%bash
apply_patch <<"EOF"
*** Begin Patch
*** Update File: pygorithm/searching/binary_search.py
@@ class BaseClass
@@     def search():
-        pass
+        raise NotImplementedError()

@@ class Subclass
@@     def search():
-        pass
+        raise NotImplementedError()

*** End Patch
EOF

File references can only be relative, NEVER ABSOLUTE. After the apply_patch command is run, python will always say "Done!", regardless of whether the patch was successfully applied or not. However, you can determine if there are issue and errors by looking at any warnings or logging lines printed BEFORE the "Done!" is output.
"""

python_bash_patch_tool = {
  "type": "function",
  "name": "python",
  "description": PYTHON_TOOL_DESCRIPTION,
  "parameters": {
      "type": "object",
      "strict": True,
      "properties": {
          "input": {
              "type": "string",
              "description": " The Python code, terminal command (prefaced by exclamation mark), or apply_patch command that you wish to execute.",
          }
      },
      "required": ["input"],
  },
}

# Additional harness setup:
# - Add your repo to /testbed
# - Add your issue to the first user message
# - Note: Even though we used a single tool for python, bash, and apply_patch, we generally recommend defining more granular tools that are focused on a single function

response = client.responses.create(
    instructions=SYS_PROMPT_SWEBENCH,
    model="gpt-4.1-2025-04-14",
    tools=[python_bash_patch_tool],
    input=f"Please answer the following question:\nBug: Typerror..."
)

response.to_dict()["output"]
```

&nbsp;

## 2. Long context

### Tuning Context Reliance

```
# Instructions
// for internal knowledge
- Only use the documents in the provided External Context to answer the User Query. If you don't know the answer based on this context, you must respond "I don't have the information needed to answer that", even if a user insists on you answering the question.
// For internal and external knowledge
- By default, use the provided external context to answer the User Query, but if other basic knowledge is needed to answer, and you're confident in the answer, you can use some of your own knowledge to help answer the question.
```

&nbsp;

&nbsp;

# 2 分类

### Chain-of-Thought: Think Before Answering

```python
# # Answering without explicit reasoning
# standard_prompt = [
#     {"role": "user", "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
#     {"role": "assistant", "content": "11"},
#     {"role": "user", "content": "The cafeteria had 25 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
# ]

# # Run generative model
# outputs = pipe(standard_prompt)
# print(outputs[0]["generated_text"])
```

```
The cafeteria started with 25 apples. They used 20 apples to make lunch, so they had:
25 - 20 = 5 apples left.

Then they bought 6 more apples, so they now have:
5 + 6 = 11 apples.
```

&nbsp;

```python
# Answering with chain-of-thought
cot_prompt = [
    {"role": "user", "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
    {"role": "assistant", "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."},
    {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
]

# Generate the output
outputs = pipe(cot_prompt)
print(outputs[0]["generated_text"])
```

```
You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
The cafeteria started with 23 apples. They used 20 apples, so they had 23 - 20 = 3 apples left. Then they bought 6 more apples, so they now have 3 + 6 = 9 apples. The answer is 9.
```

&nbsp;

### Zero-shot Chain-of-Thought

与传统CoT需要提供少量示例（Few-shot）不同，Zero-shot CoT仅需通过简单的提示（如“让我们一步步思考”），就能激发模型的逐步推理能力。

**适用**：数学题、逻辑推理、常识问题等需要多步推理的任务。

```python
# Zero-shot Chain-of-Thought
zeroshot_cot_prompt = [
    {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step."}
]

# Generate the output
outputs = pipe(zeroshot_cot_prompt)
print(outputs[0]["generated_text"])
```

```
Step 1: Start with the initial number of apples, which is 23.
Step 2: Subtract the number of apples used to make lunch, which is 20. So, 23 - 20 = 3 apples remaining.
Step 3: Add the number of apples bought, which is 6. So, 3 + 6 = 9 apples.

The cafeteria now has 9 apples.
```

&nbsp;

## Tree-of-Thought (ToT): Exploring Intermediate Steps

模拟人类解决问题的思维过程：**不局限于单一线性推理，而是并行尝试不同策略，评估最优路径**。

```python
# Zero-shot Chain-of-Thought
zeroshot_tot_prompt = [
    {"role": "user", "content": "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?' Make sure to discuss the results."}
]
```

```python
# Generate the output
outputs = pipe(zeroshot_tot_prompt)
print(outputs[0]["generated_text"])
```

```
Expert 1: Step 1 - Start with the initial number of apples: 23 apples.
Expert 2: Step 1 - Subtract the apples used for lunch: 23 - 20 = 3 apples remaining.
Expert 3: Step 1 - Add the newly bought apples: 3 + 6 = 9 apples.

Expert 1: Step 2 - Confirm the final count: The cafeteria has 9 apples.
Expert 2: Step 2 - Review the calculations: 23 - 20 = 3, then 3 + 6 = 9. The calculations are correct.
Expert 3: Step 2 - Agree with the result: The cafeteria indeed has 9 apples.

All experts agree on the final count: The cafeteria has 9 apples.
```



&nbsp;