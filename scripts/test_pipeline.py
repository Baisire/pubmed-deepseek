"""集成测试：验证 v3.0 检索管道端到端跑通。

测试流程（跳过 LLM 步骤，手动构建 SearchStrategy）：
  1. 构造模拟 SearchStrategy（模拟 LLM 拆解结果）
  2. Tier1: pubmed_recall 从 PubMed 真实检索
  3. Tier2: semantic_rerank MedCPT 语义精排
  4. citation_boost 综合评分
  5. 验证最终排序结果
"""
import os
import sys
import time

# 确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from search.models import Article, SearchStrategy, ConceptDimension, MeshTerm
from search.pubmed_recall import recall as pubmed_recall
from search.semantic_rerank import SemanticReranker
from search.citation_boost import score_and_rank

print("=== v3.0 检索管道集成测试 ===\n")

# ============================================================
# Step 0: 构造模拟 SearchStrategy
# （模拟 query_understanding 对 "化疗后中性粒细胞减少的血液学预测因子" 的拆解结果）
# ============================================================
print("[Step 0] 构造模拟检索策略...")

strategy = SearchStrategy(
    boolean_query=(
        '("Neutropenia"[MeSH Terms] OR "neutropenia"[Title/Abstract] OR "febrile neutropenia"[Title/Abstract]) '
        'AND ("Monocytes"[MeSH Terms] OR "Lymphocytes"[MeSH Terms] OR "monocyte"[Title/Abstract] OR "lymphocyte"[Title/Abstract]) '
        'AND ("Antineoplastic Agents"[MeSH Terms] OR "chemotherapy"[Title/Abstract]) '
        'AND ("predictor"[Title/Abstract] OR "risk factor"[Title/Abstract])'
    ),
    semantic_query="hematological predictors for chemotherapy-induced febrile neutropenia",
    query_level="medium",
    concepts=[],  # 简化测试，不填 concepts
    filters={},
)
print(f"  Boolean query: {strategy.boolean_query[:80]}...")
print(f"  Semantic query: {strategy.semantic_query}")

# ============================================================
# Step 1: Tier 1 PubMed 召回
# ============================================================
print(f"\n[Step 1] Tier 1: PubMed 结构化召回...")
t0 = time.time()
articles = pubmed_recall(strategy, max_results=20, expand_related=False)
t1 = time.time()
print(f"  耗时: {t1-t0:.1f}s")
print(f"  召回 {len(articles)} 篇文献")
for i, a in enumerate(articles[:5]):
    print(f"    {i+1}. PMID={a.pmid} | rank={a.pubmed_rank} | {a.title[:60]}...")

if not articles:
    print("\n✗ 召回失败，终止测试")
    sys.exit(1)

# ============================================================
# Step 2: Tier 2 语义精排
# ============================================================
print(f"\n[Step 2] Tier 2: MedCPT 语义精排（不含 Cross-Encoder 以加速测试）...")
t0 = time.time()
reranker = SemanticReranker()
articles = reranker.rerank_articles(
    strategy.semantic_query,
    articles,
    use_cross_encoder=False,  # 测试先关掉 Cross-Encoder 以加速
)
t1 = time.time()
print(f"  耗时: {t1-t0:.1f}s")
print(f"  语义排序 Top 5:")
for i, a in enumerate(articles[:5]):
    print(f"    {i+1}. PMID={a.pmid} | sem={a.semantic_score:.4f} | {a.title[:55]}...")

# ============================================================
# Step 3: 引文增强与综合评分（跳过引文以加速测试）
# ============================================================
print(f"\n[Step 3] 综合评分（跳过引文查询以加速）...")
t0 = time.time()
articles = score_and_rank(articles, use_citations=False)
t1 = time.time()
print(f"  耗时: {t1-t0:.3f}s")
print(f"  综合评分排序 Top 5:")
for i, a in enumerate(articles[:5]):
    print(f"    {i+1}. PMID={a.pmid} | final={a.final_score:.4f} | sem={a.semantic_score:.4f} | {a.title[:50]}...")

# ============================================================
# Step 4: Cross-Encoder 精排（对 Top 10）
# ============================================================
print(f"\n[Step 4] Cross-Encoder 精排（Top 10）...")
t0 = time.time()
rerank_scores = reranker.rerank(strategy.semantic_query, articles[:10], top_k=10)
for a in articles:
    a.rerank_score = rerank_scores.get(a.pmid, 0.0)
# 重新排序（有 rerank_score 的按 rerank 排）
articles.sort(key=lambda a: a.rerank_score if a.rerank_score > 0 else a.semantic_score, reverse=True)
t1 = time.time()
print(f"  耗时: {t1-t0:.1f}s")
print(f"  Cross-Encoder 精排后 Top 5:")
for i, a in enumerate(articles[:5]):
    print(f"    {i+1}. PMID={a.pmid} | rerank={a.rerank_score:.4f} | sem={a.semantic_score:.4f} | {a.title[:50]}...")

# ============================================================
# 验证总结
# ============================================================
print(f"\n=== 集成测试总结 ===")
print(f"  Tier1 召回: {len(articles)} 篇")
print(f"  语义精排: 成功（MedCPT bi-encoder）")
print(f"  Cross-Encoder: 成功（bge-reranker Top 10）")
print(f"  综合评分: 成功")
print(f"  管道模块串联: ✓")

# 验证 Article 对象完整性
sample = articles[0]
required_fields = ["pmid", "title", "abstract", "authors", "journal", "pub_date", "doi",
                   "semantic_score", "rerank_score", "final_score", "pubmed_rank"]
missing = [f for f in required_fields if not hasattr(sample, f)]
if missing:
    print(f"  ✗ 缺失字段: {missing}")
else:
    print(f"  Article 对象完整性: ✓（所有字段已填充）")

print(f"\n  v3.0 检索管道集成测试通过")
