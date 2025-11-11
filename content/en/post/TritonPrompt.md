---
date: 2025-07-27T11:00:59-04:00
description: ""
featured_image: "/images/TritonPrompt/jaz.png"
tags: ["paper"]
title: "Triton Prompt"
---

# AutoTriton

### TritonBench Infer Prompt

~~~python
SYS_INSTRUCTION = """Use triton language write a kernel and wrapper according to the following instruction:
"""
INSTRUCTION_EXTRA = """The wrapper function should have same input and output as in instruction, and written with 'def xxx' DIRECTLY, do not wrap the wrapper inside a class. You may write it as:
```python
@triton.jit
def kernel([parameters]):
	# your implementation

def wrapper ([parameters]):
	# your implementation
```
"""
prompt = f"""{SYS_INSTRUCTION}
{ORIGINAL_INSTRUCTION}
{INSTRUCTION_EXTRA}
"""
~~~

### KernelBench Infer Prompt

~~~python
PROBLEM_STATEMENT = """You are given a pytorch function, and your task is to write the same triton implementation for it.
The triton implementation should change the name from Model to ModelNew, and have same input and output as the pytorch function."""
PROBLEM_INSTRUCTION = """Optimize the architecture with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no input and init function, no other text, and NO testing code! **Remember to Name your optimized output architecture ModelNew, do not use Model again!**"""
prompt = f"""{PROBLEM_STATEMENT}
{PROBLEM_INSTRUCTION}
	Now, you need to write the triton implementation for the following pytorch code:
	```
	{arc_src}
	```
"""
~~~

### 



# CUDA- L1

### SFT

#### Task for CUDA Optimization

```python
You are an expert in CUDA programming and GPU kernel optimization. Now you’re tasked with developing a
high-performance cuda implementation of Softmax. The implementation must:
• Produce identical results to the reference PyTorch implementation.
• Demonstrate speed improvements on GPU.
• Maintain stability for large input values.
```

#### Reference Implementation (exact copy)

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).
        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
```



### RL



<!--more-->

&nbsp;