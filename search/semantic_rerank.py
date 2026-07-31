"""语义精排 - MedCPT bi-encoder + bge-reranker cross-encoder 两段式精排。

职责：
- Tier 2.1: MedCPT 编码查询与文献，计算 cosine 语义相似度（bi-encoder，快）
- Tier 2.2: bge-reranker 对 Top-K 偙选做 cross-encoder 精排（慢但准）

设计要点：
- 模型懒加载（首次调用时加载，之后单例复用）
- 文献向量优先查 LanceDB 缓存，未命中才编码
- Cross-Encoder 仅对 Top 10-15 计算（实测 ~1s/篇 CPU）
- 任一模型加载失败时优雅降级

环境配置：
- 国内需设置 HF_ENDPOINT=https://hf-mirror.com 解决 HuggingFace 连接超时
- 模型缓存目录设为项目 data/ 目录，避免 C 盘生成文件
"""

import os
import time
from typing import Optional

import numpy as np

from .models import Article
from .embedding_store import EmbeddingStore

# 环境配置（必须在 import sentence_transformers 之前设置）
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.environ.setdefault("HF_HOME", os.path.join(_DATA_DIR, "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 模型名称
_MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
_MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Cross-Encoder 最大候选数（实测 ~1s/篇 CPU，15 篇约 15s）
_MAX_RERANK_CANDIDATES = 15


class SemanticReranker:
    """两段式语义精排器：MedCPT 相似度 + bge-reranker 重排序。"""

    def __init__(self, device: str = "cpu",
                 embedding_store: Optional[EmbeddingStore] = None) -> None:
        self._device = device
        self._query_encoder = None
        self._article_encoder = None
        self._reranker = None
        self._store = embedding_store or EmbeddingStore()

    # ------------------------------------------------------------------
    # 模型懒加载
    # ------------------------------------------------------------------

    def _load_query_encoder(self):
        if self._query_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._query_encoder = SentenceTransformer(_MEDCPT_QUERY_MODEL, device=self._device)
        return self._query_encoder

    def _load_article_encoder(self):
        if self._article_encoder is None:
            from sentence_transformers import SentenceTransformer
            self._article_encoder = SentenceTransformer(_MEDCPT_ARTICLE_MODEL, device=self._device)
        return self._article_encoder

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(_RERANKER_MODEL, max_length=512, device=self._device)
        return self._reranker

    # ------------------------------------------------------------------
    # Tier 2.1: MedCPT bi-encoder 语义相似度
    # ------------------------------------------------------------------

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本为 768d 向量（归一化）。"""
        encoder = self._load_query_encoder()
        vec = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0]

    def encode_articles(self, articles: list[Article]) -> dict[str, np.ndarray]:
        """编码文献摘要为向量，优先查缓存。

        返回 {pmid: vector}，向量已归一化。
        """
        if not articles:
            return {}

        # 1. 查缓存
        pmids = [a.pmid for a in articles]
        cached = self._store.get_vectors_batch(pmids)
        result = dict(cached)

        # 2. 编码未命中的
        uncached = [a for a in articles if a.pmid not in result]
        if uncached:
            encoder = self._load_article_encoder()
            texts = []
            for a in uncached:
                # MedCPT Article Encoder 期望 title + abstract 拼接
                text = a.title if a.abstract == "" else f"{a.title}. {a.abstract}"
                texts.append(text)

            vectors = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            for a, vec in zip(uncached, vectors):
                result[a.pmid] = vec
                a.embedding_cached = True
                # 写入缓存
                pub_year = 0
                try:
                    pub_year = int(a.pub_date[:4]) if a.pub_date[:4].isdigit() else 0
                except (ValueError, IndexError):
                    pass
                try:
                    self._store.store(a.pmid, a.title, a.abstract, vec,
                                      journal=a.journal, pub_year=pub_year)
                except Exception:
                    pass  # 缓存写入失败不影响主流程

        return result

    def rank_by_similarity(self, query_vec: np.ndarray,
                           article_vecs: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        """计算查询与各文献的 cosine 相似度，返回按相似度降序的 (pmid, score) 列表。"""
        scores = []
        for pmid, vec in article_vecs.items():
            sim = float(np.dot(query_vec, vec))  # 已归一化，点积即 cosine
            scores.append((pmid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ------------------------------------------------------------------
    # Tier 2.2: bge-reranker cross-encoder 精排
    # ------------------------------------------------------------------

    def rerank(self, query: str, articles: list[Article],
               top_k: int = _MAX_RERANK_CANDIDATES) -> dict[str, float]:
        """对 Top-K 候选做 Cross-Encoder 精排。

        Args:
            query: 语义查询文本
            articles: 候选文献列表（应已按 bi-encoder 相似度排序）
            top_k: 送入 reranker 的最大候选数

        Returns:
            {pmid: rerank_score} 字典
        """
        candidates = articles[:top_k]
        if not candidates:
            return {}

        try:
            reranker = self._load_reranker()
        except Exception:
            # reranker 加载失败，返回空（调用方用 bi-encoder 分数兜底）
            return {}

        pairs = []
        for a in candidates:
            text = a.title if a.abstract == "" else f"{a.title}. {a.abstract}"
            pairs.append((query, text))

        try:
            scores = reranker.predict(pairs)
            return {a.pmid: float(s) for a, s in zip(candidates, scores)}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # 完整两段式精排
    # ------------------------------------------------------------------

    def rerank_articles(self, query: str, articles: list[Article],
                        use_cross_encoder: bool = True) -> list[Article]:
        """完整的两段式语义精排：bi-encoder 召回排序 + cross-encoder 精排。

        Args:
            query: 英文语义查询文本
            articles: 候选文献列表
            use_cross_encoder: 是否启用 Cross-Encoder 精排（可关闭以降低延迟）

        Returns:
            排序后的 Article 列表（semantic_score 和 rerank_score 已填充）
        """
        if not articles:
            return []

        # Tier 2.1: bi-encoder 语义相似度
        query_vec = self.encode_query(query)
        article_vecs = self.encode_articles(articles)

        sim_scores = self.rank_by_similarity(query_vec, article_vecs)
        sim_map = dict(sim_scores)

        for a in articles:
            a.semantic_score = sim_map.get(a.pmid, 0.0)

        # 按 bi-encoder 相似度排序
        articles.sort(key=lambda a: a.semantic_score, reverse=True)

        # Tier 2.2: cross-encoder 精排（可选）
        if use_cross_encoder and len(articles) > 0:
            rerank_map = self.rerank(query, articles)
            if rerank_map:
                for a in articles:
                    a.rerank_score = rerank_map.get(a.pmid, 0.0)
                # 有 rerank 分数的按 rerank 排序，没有的留在后面
                articles.sort(key=lambda a: a.rerank_score, reverse=True)

        return articles
