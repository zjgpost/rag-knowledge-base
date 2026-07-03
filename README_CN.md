# RAG Knowledge Base 中文文档

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个企业级 **混合检索** 知识库与智能问答系统。多语言意图级评测：NDCG@10=0.855 / Recall@10=0.973。

> 对应简历项目：*企业级 RAG 知识库与智能问答系统*  
> GitHub：https://github.com/zjgpost/rag-knowledge-base

---

## ✨ 核心亮点

- **三阶段检索**：Dense（语义召回）+ Sparse（BM25 关键词召回）+ Rerank（精排）
- **语义缓存**：基于 Embedding 相似度缓存答案，降低 LLM API 调用量
- **RBAC 元数据过滤**：文档级与内容级权限隔离
- **自适应分块**：固定分块 / 语义分块 / 递归分块，按文档类型自动选择
- **可复现评测**：`benchmarks/run_real_retrieval_eval.py` 一键输出 NDCG@10 与 Recall@10

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                         用户问题                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  语义缓存层                                                  │
│  - Embedding 相似度 > 阈值 → 直接返回缓存答案                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  第一阶段：Dense Retrieval（TF-IDF / BGE / 多语言 Embedding）│
│  第二阶段：Sparse Retrieval（BM25）                          │
│  第三阶段：Reranker（Cross-Encoder 精排）                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RBAC 元数据过滤                                             │
│  - 部门 / 角色 / 访问级别校验                                │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  答案生成（Answer Synthesis）                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

```bash
git clone https://github.com/zjgpost/rag-knowledge-base.git
cd rag-knowledge-base
pip install -r requirements.txt

# 检查哪些模型已下载 / 缺失
python scripts/check_models.py

# 下载缺失模型（只需一次；模型存放在 models/ 目录，不会被 git 提交）
python scripts/download_models.py --missing

# 运行基础示例
python examples/basic_rag.py

# 运行检索评测
python benchmarks/run_retrieval_eval.py

# 运行测试
python -m pytest tests -v
```

---

## 📊 评测结果

### 示例数据集

在 20 个查询的示例数据集上：

| 方法        | NDCG@10 | Recall@10 |
|-------------|--------:|----------:|
| Dense-only  | 0.908   | 1.000     |
| Sparse-only | 0.908   | 1.000     |
| Hybrid      | 0.926   | 1.000     |

### 真实电商客服数据集

基于公开数据集 [SEA e-commerce customer support sample](https://huggingface.co/datasets/nwchang/sea-ecommerce-customer-support-sample)（1000 条真实客服回复，100 条查询）：

| 方法        | NDCG@10 | Recall@10 |
|-------------|--------:|----------:|
| Dense-only  | 0.346   | 0.500     |
| Sparse-only | 0.318   | 0.460     |
| Hybrid      | 0.352   | 0.520     |

### 多语言意图级评测

在 15 个意图文档、75 条真实多语言客服查询上，使用 `paraphrase-multilingual-MiniLM-L12-v2` 做 dense 召回，`Alibaba-NLP/gte-multilingual-reranker-base` 做 rerank，dense/sparse 权重 0.9/0.1：

| 方法        | NDCG@10 | Recall@10 |
|-------------|--------:|----------:|
| Dense-only  | 0.867   | 0.973     |
| Sparse-only | 0.518   | 0.600     |
| Hybrid      | 0.855   | 0.973     |

本地复现：

```bash
python benchmarks/run_real_retrieval_eval.py \
  --use-st --use-cross-encoder \
  --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
  --reranker-model Alibaba-NLP/gte-multilingual-reranker-base \
  --dense-weight 0.9 --sparse-weight 0.1 \
  --corpus benchmarks/dataset/ecommerce_intent_corpus.jsonl \
  --queries benchmarks/dataset/ecommerce_queries.jsonl
```

> PRF 查询扩展也做了实验，但在此语料上 NDCG@10 从 0.855 降至 0.834，因此作为可选开关（`--use-query-expansion`）保留，而非默认开启。

---

## 🔍 使用示例

```python
from kb import KnowledgeBase

kb = KnowledgeBase()
kb.add_documents(
    ["Redis connection pool max_connections ...", "Redis performance optimization ..."],
    metadata=[
        {"department": "engineering", "access_level": "internal"},
        {},
    ],
    chunk_strategy="semantic",
)

result = kb.query(
    "Redis connection pool full",
    user_role={"department": "engineering", "clearance": "internal"},
)
print(result["answer"])
```

> 注意：当前 `KnowledgeBase` 的答案生成是把检索到的 top 文档拼接返回。若要生成自然语言答案，可把检索结果传入你自己的 LLM prompt。

---

## 🧰 模型管理

检查本地已下载/缺失模型：

```bash
python scripts/check_models.py
```

只下载缺失模型：

```bash
python scripts/download_models.py --missing
```

下载指定模型：

```bash
python scripts/download_models.py --model paraphrase-multilingual-MiniLM-L12-v2
```

列出所有配置模型：

```bash
python scripts/download_models.py --list
```

模型存放在 `models/` 目录，已被 git 忽略。

---

## 📚 添加自己的文档

```python
from kb import KnowledgeBase

kb = KnowledgeBase(dense_weight=0.9, sparse_weight=0.1)

kb.add_documents(
    documents=[
        "你的第一篇文档...",
        "你的第二篇文档...",
    ],
    metadata=[
        {"department": "engineering", "access_level": "internal"},
        {"department": "operations", "access_level": "public"},
    ],
    chunk_strategy="semantic",  # 或 "fixed" / "recursive"
)

result = kb.query(
    "你的问题？",
    user_role={"department": "engineering", "clearance": "internal"},
)
print(result["answer"])
```

---

## ⚖️ 调优 Hybrid 权重

使用项目内置脚本在自定义语料上搜索最优 dense/sparse 权重：

```bash
python scripts/tune_hybrid.py \
  --corpus benchmarks/dataset/ecommerce_intent_corpus.jsonl \
  --queries benchmarks/dataset/ecommerce_queries.jsonl \
  --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
  --use-st
```

脚本会输出多组权重下的 NDCG@10 与 Recall@10。

---

## 🔁 查询扩展（可选）

项目实现了伪相关反馈（PRF）查询扩展，但**默认不开启**，因为在 intent-level 语料上 NDCG@10 从 0.855 降至 0.834。可实验性开启：

```bash
python benchmarks/run_real_retrieval_eval.py \
  --use-st --use-cross-encoder \
  --dense-model paraphrase-multilingual-MiniLM-L12-v2 \
  --reranker-model Alibaba-NLP/gte-multilingual-reranker-base \
  --use-query-expansion \
  --expansion-terms 5
```

---

## 🧪 测试

```bash
python -m pytest tests -v
```

当前状态：**9 个测试全部通过**。

---

## 📝 相关技术博客

- 《[RAG 检索不准？试试多语言三阶段混合检索：Dense + Sparse + Rerank](https://juejin.cn/spost/7657863114353426470)》（掘金）  
  [CSDN 镜像](https://blog.csdn.net/janguo_qql/article/details/162549098?sharetype=blogdetail&sharerId=162549098&sharerefer=PC&sharesource=janguo_qql&spm=1011.2480.3001.8118)
- （待发布）《语义缓存：被忽视的 LLM 成本优化点》
- （待发布）《企业级知识库的 RBAC 权限隔离设计》

- 《企业级知识库的 RBAC 权限隔离设计》

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE)。

---

## 🤝 贡献

欢迎提交 Issue 和 PR。新增检索策略或缓存策略时，请同步补充测试。
