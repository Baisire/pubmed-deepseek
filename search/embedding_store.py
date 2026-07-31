"""LanceDB 向量存储 - 文献 embedding 的持久化与检索。

职责：
- 将文献摘要的 MedCPT 向量持久化到 LanceDB
- 按 PMID 查询已缓存的向量（避免重复编码）
- 向量相似度检索（cosine）

设计要点：
- 嵌入式数据库，无需独立服务进程
- 向量永久有效（摘要不变则向量不变）
- 首次编码后写入缓存，后续检索命中缓存 ~1ms/篇
"""

import os
from typing import Optional

import numpy as np

# LanceDB 数据目录（放 D 盘项目目录下，避免 C 盘生成文件）
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_LANCEDB_PATH = os.path.join(_DATA_DIR, "lancedb")

_TABLE_NAME = "article_vectors"
_VECTOR_DIM = 768  # MedCPT 输出维度


class EmbeddingStore:
    """文献向量存储，封装 LanceDB 读写。"""

    def __init__(self, db_path: str = _LANCEDB_PATH) -> None:
        os.makedirs(db_path, exist_ok=True)
        import lancedb
        self._db = lancedb.connect(db_path)
        self._table = None

    def _ensure_table(self, vector_dim: int = _VECTOR_DIM) -> None:
        """确保表存在，首次调用时创建。"""
        if self._table is not None:
            return
        existing = self._db.list_tables()
        if _TABLE_NAME in existing:
            self._table = self._db.open_table(_TABLE_NAME)
        else:
            # 用空 schema 创建表，首次写入时自动推断
            self._table = None  # 延迟到首次写入

    def store(self, pmid: str, title: str, abstract: str,
              vector: np.ndarray, journal: str = "", pub_year: int = 0) -> None:
        """存储一篇文献的向量。如果 PMID 已存在则跳过。"""
        self._ensure_table()
        record = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "vector": vector.tolist(),
            "journal": journal,
            "pub_year": pub_year,
        }
        if self._table is None:
            # 首次写入，创建表
            self._table = self._db.create_table(_TABLE_NAME, [record])
        else:
            # 检查是否已存在（避免重复写入）
            existing = self.get_vector(pmid)
            if existing is not None:
                return
            self._table.add([record])

    def get_vector(self, pmid: str) -> Optional[np.ndarray]:
        """获取已缓存的文献向量，未命中返回 None。"""
        self._ensure_table()
        if self._table is None:
            return None
        try:
            results = self._table.search().where(f"pmid = '{pmid}'").limit(1).to_list()
            if results:
                return np.array(results[0]["vector"], dtype=np.float32)
        except Exception:
            pass
        return None

    def get_vectors_batch(self, pmids: list[str]) -> dict[str, np.ndarray]:
        """批量获取已缓存的向量，返回 {pmid: vector}（仅含命中的）。"""
        result: dict[str, np.ndarray] = {}
        for pmid in pmids:
            vec = self.get_vector(pmid)
            if vec is not None:
                result[pmid] = vec
        return result

    def search(self, query_vector: np.ndarray, limit: int = 20) -> list[dict]:
        """向量相似度检索，返回 Top-K 文献（含距离）。"""
        self._ensure_table()
        if self._table is None:
            return []
        try:
            results = self._table.search(query_vector.tolist()).limit(limit).to_list()
            # LanceDB 返回 _distance（L2 距离），转为相似度
            for r in results:
                r["similarity"] = 1.0 - r.get("_distance", 0.0)
            return results
        except Exception:
            return []

    def count(self) -> int:
        """返回已缓存的文献向量总数。"""
        self._ensure_table()
        if self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0
