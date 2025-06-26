---
date: 2025-06-23T10:00:59-04:00
description: ""
featured_image: "/images/langchain-textSplitter/jaz.png"
tags: ["langchain", LLM"]
title: "langchain - text_splitter"
---

```python
import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()
```

<!--more-->

```python
# Define the embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # Update to a valid embedding model if needed


# Function to create and persist vector store
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")
```

&nbsp;

###  基于字符的分割

```python
# 1. Character-based Splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")
```

+ 特点：按固定字符数分割，不考虑文本结构。

+ 适用场景：当需要严格控制每个分块的大小（以字符数为单位）时，且文本结构不重要或没有明显结构（如代码、日志文件）。这种分割方式速度快，但可能破坏句子或单词的完整性。

### 基于句子的分割

```python
# 2. Sentence-based Splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")
```

+ 特点：按句子边界分割，确保每个分块包含完整的句子。
+ 适用场景：对语义连贯性要求高的任务，如机器翻译、文本摘要。它保持了句子完整性，但分块大小可能不均匀（长句子可能导致分块过大）。

### 基于标记的分割

```python
# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")
```

+  特点：使用分词器（如GPT-2的tokenizer）按标记（token）分割。
+ 适用场景：需要与特定Transformer模型（如GPT系列、BERT）配合使用时，因为这些模型有严格的标记数量限制。可确保分块后标记数不超过模型输入限制。

### 递归的基于字符的分割

```python
# 4. Recursive Character-based Splitting: most people use
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")
```

+ 特点：在字符数限制内，优先按自然边界（如段落、句子）递归分割。
+ 适用场景：通用文本处理，平衡分块大小和语义完整性。它是大多数情况下的推荐选择，尤其当文本有层次结构（如文章包含段落和句子）时。

### 自定义分割

```python
# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n--- Using Custom Splitting ---")
```

+ 适用场景：处理非标准结构的文档（如特定格式的报告、表格、代码文件），或当现有分割方法无法满足需求时。

&nbsp;

```python
class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Custom logic for splitting text
        return text.split("\n\n")  # Example: split by paragraphs

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")


# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory, embedding_function=embeddings
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "How did Juliet die?"

# Query each vector store
query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
query_vector_store("chroma_db_custom", query)
```

