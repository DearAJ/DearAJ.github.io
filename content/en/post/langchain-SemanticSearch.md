---
date: 2025-06-24T10:00:59-04:00
description: ""
featured_image: "/images/langchain-SemanticSearch/jaz.png"
tags: ["langchain", "LLM"]
title: "langchain - 混合搜索"
---

先通过BM25快速筛选关键字，再用Reranker对候选文档进行精细排序。

```python
def keyword_and_reranking_search(query, top_k=3, num_candidates=10):
    print("Input question:", query)

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]		# 选取分数最高的 num_candidates 个文档
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    print(f"Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

    
    #Add re-ranking
    docs = [texts[hit['corpus_id']] for hit in bm25_hits]

    print(f"\nTop-3 hits by rank-API ({len(bm25_hits)} BM25 hits re-ranked)")
    results = co.rerank(query=query, documents=docs, top_n=top_k, return_documents=True)
    for hit in results.results:
        print("\t{:.3f}\t{}".format(hit.relevance_score, hit.document.text.replace("\n", " ")))
```

+ #### bm25

  基于词频和逆文档频率，计算每个文档与查询的关键词匹配分数。

  + **词频 (TF)**: 这是查询中的词 q_i在文档 D 中出现的频率。词频是衡量一个词在文档中重要性的基本指标。词频越高，这个词在文档中的重要性通常越大。
  + **逆文档频率 (IDF)**: 逆文档频率是衡量一个词对于整个文档集合的独特性或信息量的指标。它是由整个文档集合中包含该词的文档数量决定的。一个词在很多文档中出现，其IDF值就会低，反之则高。这意味着罕见的词通常有更高的IDF值，从而在相关性评分中拥有更大的权重。

+ #### Reranker

  使用语义模型（Cohere的Rerank API）重新评分，会考虑上下文语义（而不仅是关键词），返回更相关的排序。

  Reranker的计算成本较高，因此仅对BM25筛选后的少量候选（`num_candidates`）进行处理。

&nbsp;

