---
date: 2026-06-29T10:00:59-04:00
description: ""
featured_image: "/images/multimodal/jaz.png"
tags: ["multimodal", "LLM"]
title: "multimodal learning"
---

## CLIP



### **CLIP 文本-图像匹配**

#### **(1) 使用 `CLIPProcessor` 预处理**

```python
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"

# Load a tokenizer to preprocess the text
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

# Load a processor to preprocess the images
clip_processor = CLIPProcessor.from_pretrained(model_id)

# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(model_id)


# 处理文本
text = ["a photo of a cat"]
inputs = processor(text=text, return_tensors="pt", padding=True)  # -> {"input_ids": ..., "attention_mask": ...}

# 处理图像
image = Image.open("cat.jpg")
processed_image = processor(images=image, return_tensors="pt")["pixel_values"]  # -> [1, 3, 224, 224]
```

#### **(2) 生成 Embedding**

```python
# 文本嵌入
text_embedding = model.get_text_features(**inputs)  # -> [1, 512]

# 图像嵌入
image_embedding = model.get_image_features(processed_image)  # -> [1, 512]
```

#### **(3) 计算相似度**

```python
# 归一化（使点积=余弦相似度）
text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

# 计算相似度
similarity = (text_embedding @ image_embedding.T).item()  # 值在 [-1, 1] 之间
```

**为什么 `CLIPProcessor` 后还需要 Embedding？**

| 步骤              | 作用                               | 输入                        | 输出                        |
| :---------------- | :--------------------------------- | :-------------------------- | :-------------------------- |
| **CLIPProcessor** | 预处理（文本→token，图像→张量）    | 原始文本/图像               | `input_ids`, `pixel_values` |
| **Embedding**     | 生成语义向量（文本/图像→高维向量） | `input_ids`, `pixel_values` | `[1, 512]` 向量             |

- **`CLIPProcessor`** 只是 **数据预处理**，不涉及模型推理。
- **Embedding** 是 **模型的核心计算**，将输入映射到语义空间。



