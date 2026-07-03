# RAG Knowledge Base

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade **hybrid retrieval** knowledge base and Q&A system. Multilingual intent-level benchmark: NDCG@10=0.855 / Recall@10=0.973.

> 📖 [中文文档](README_CN.md)  
> Corresponds to the resume project: *Enterprise RAG Knowledge Base and Intelligent Q&A System*  
> GitHub: https://github.com/zjgpost/rag-knowledge-base

---

## ✨ Highlights

- **Three-stage retrieval**: Dense (TF-IDF/BGE-style) + Sparse (BM25) + Rerank.
- **Semantic cache**: answers cached by embedding similarity to reduce LLM API calls.
- **RBAC metadata filtering**: document-level and content-level permission control.
- **Adaptive chunking**: fixed / semantic / recursive strategies chosen by document type.
- **Reproducible benchmark**: `benchmarks/run_retrieval_eval.py` outputs NDCG@10 and Recall@10.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Question                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Semantic Cache                                             │
│  - Embedding similarity > threshold → return cached answer   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Dense Retrieval (TF-IDF / BGE)                    │
│  Stage 2: Sparse Retrieval (BM25)                           │
│  Stage 3: Reranker                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RBAC Metadata Filter                                       │
│  - Department / clearance / access_level checks             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Answer Synthesis                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/zjgpost/rag-knowledge-base.git
cd rag-knowledge-base
pip install -r requirements.txt

# Check which models are already present / missing
python scripts/check_models.py

# Download missing models (only needed once; models are stored in models/ and ignored by git)
python scripts/download_models.py --missing

# Run the basic RAG example
python examples/basic_rag.py

# Run the benchmark
python benchmarks/run_retrieval_eval.py

# Run tests
python -m pytest tests -v
```

---

## 📊 Benchmark Results

Run on the built-in example dataset (20 queries, keyword-precise + semantic + hybrid-advantage):

| Method       | NDCG@10 | Recall@10 |
|--------------|--------:|----------:|
| Dense-only   | 0.908   | 1.000     |
| Sparse-only  | 0.908   | 1.000     |
| Hybrid       | 0.926   | 1.000     |

**Hybrid vs Dense NDCG@10 improvement: +2.0%**

### Real-world e-commerce benchmark

We also provide a reproducible benchmark built from the public [SEA e-commerce customer support sample](https://huggingface.co/datasets/nwchang/sea-ecommerce-customer-support-sample) dataset (1000 real support replies, 100 customer queries):

| Method       | NDCG@10 | Recall@10 |
|--------------|--------:|----------:|
| Dense-only   | 0.346   | 0.500     |
| Sparse-only  | 0.318   | 0.460     |
| Hybrid       | 0.352   | 0.520     |

**Hybrid vs Dense NDCG@10 improvement: +1.8%**  
**Hybrid vs Sparse NDCG@10 improvement: +10.6%**

Run it locally:

```bash
python scripts/build_reply_benchmark.py   # download dataset and build corpus/queries
python benchmarks/run_real_retrieval_eval.py
```

> The real-world numbers above use TF-IDF for dense retrieval. To improve further, enable `sentence-transformers` (all-MiniLM-L6-v2):
> ```bash
> pip install sentence-transformers
> python benchmarks/run_real_retrieval_eval.py --use-st
> ```
> Neural embeddings typically push NDCG@10 into the 0.50–0.70 range on this task.

### Multilingual intent-level benchmark

We additionally provide an intent-level evaluation (15 intent documents, 75 real multilingual customer queries). Using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for dense retrieval and `Alibaba-NLP/gte-multilingual-reranker-base` for reranking with a 0.9/0.1 dense/sparse weight:

| Method       | NDCG@10 | Recall@10 |
|--------------|--------:|----------:|
| Dense-only   | 0.867   | 0.973     |
| Sparse-only  | 0.518   | 0.600     |
| Hybrid       | 0.855   | 0.973     |

Run it locally:

```bash
python benchmarks/run_real_retrieval_eval.py \
  --use-st --use-cross-encoder \
  --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
  --reranker-model Alibaba-NLP/gte-multilingual-reranker-base \
  --dense-weight 0.9 --sparse-weight 0.1 \
  --corpus benchmarks/dataset/ecommerce_intent_corpus.jsonl \
  --queries benchmarks/dataset/ecommerce_queries.jsonl
```

> PRF query expansion was also evaluated but decreased NDCG@10 on this corpus (0.855 → 0.834), so it remains an optional flag (`--use-query-expansion`) rather than the default.

---

## 🔍 Usage Example

```python
from kb import KnowledgeBase

kb = KnowledgeBase()
kb.add_documents(
    ["Redis connection pool max_connections ...", "Redis performance optimization ..."],
    metadata=[{"department": "engineering", "access_level": "internal"}, {}],
    chunk_strategy="semantic",
)

result = kb.query(
    "Redis connection pool full",
    user_role={"department": "engineering", "clearance": "internal"},
)
print(result["answer"])
```

> Note: the current `KnowledgeBase` synthesizes answers by concatenating the top retrieved documents. To generate natural-language answers, pass the retrieved documents to your own LLM prompt.

---

## 🧰 Model Management

Check which models are present locally:

```bash
python scripts/check_models.py
```

Download only missing models:

```bash
python scripts/download_models.py --missing
```

Download a specific model:

```bash
python scripts/download_models.py --model paraphrase-multilingual-MiniLM-L12-v2
```

List all configured models:

```bash
python scripts/download_models.py --list
```

Models are stored in `models/` and ignored by git.

---

## 📚 Adding Your Own Documents

```python
from kb import KnowledgeBase

kb = KnowledgeBase(dense_weight=0.9, sparse_weight=0.1)

kb.add_documents(
    documents=[
        "Your first document text...",
        "Your second document text...",
    ],
    metadata=[
        {"department": "engineering", "access_level": "internal"},
        {"department": "operations", "access_level": "public"},
    ],
    chunk_strategy="semantic",  # or "fixed" / "recursive"
)

result = kb.query(
    "Your question?",
    user_role={"department": "engineering", "clearance": "internal"},
)
print(result["answer"])
```

---

## ⚖️ Tuning Hybrid Weights

Use the included tuning script to grid-search dense/sparse weights on your own corpus:

```bash
python scripts/tune_hybrid.py \
  --corpus benchmarks/dataset/ecommerce_intent_corpus.jsonl \
  --queries benchmarks/dataset/ecommerce_queries.jsonl \
  --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
  --use-st
```

The script evaluates multiple weight combinations and prints NDCG@10 / Recall@10 for each.

---

## 🔁 Query Expansion (Optional)

Pseudo-relevance feedback (PRF) expansion is available but **not enabled by default** because it hurt NDCG@10 on the intent-level corpus (0.855 → 0.834). To experiment:

```bash
python benchmarks/run_real_retrieval_eval.py \
  --use-st --use-cross-encoder \
  --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
  --reranker-model Alibaba-NLP/gte-multilingual-reranker-base \
  --use-query-expansion \
  --expansion-terms 5
```

---

## 🧪 Tests

```bash
python -m pytest tests -v
```

Current status: **9 tests passing**.

---

## 📝 Related Blog Posts

- [RAG 检索不准？试试多语言三阶段混合检索：Dense + Sparse + Rerank](https://juejin.cn/spost/7657863114353426470)（掘金）  
  [CSDN 镜像](https://blog.csdn.net/janguo_qql/article/details/162549098?sharetype=blogdetail&sharerId=162549098&sharerefer=PC&sharesource=janguo_qql&spm=1011.2480.3001.8118)
- (Placeholder) Semantic Cache: The Overlooked LLM Cost Optimization
- (Placeholder) RBAC Design for Enterprise Knowledge Bases

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🤝 Contributing

Issues and PRs are welcome. Please add tests for new retrieval strategies or cache policies.
