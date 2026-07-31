"""快速端到端验证：pipeline.search() 实际调用。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from search.pipeline import search


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    print("=== v3.0 pipeline.search() 端到端验证 ===")
    print("查询: 化疗后中性粒细胞减少的预测因子")
    print()

    def progress(msg):
        print(f"  [进度] {msg}")

    t0 = time.time()
    result = search(
        user_input="化疗后中性粒细胞减少的预测因子",
        deepseek_api_key=api_key,
        max_results=5,
        candidate_pool_size=20,
        use_cross_encoder=False,
        use_citations=False,
        progress_callback=progress,
    )
    t1 = time.time()

    print()
    print(f"总耗时: {t1-t0:.1f}s")
    print(f"召回文献: {len(result.articles)} 篇")
    print(f"检索策略 fallback: {result.strategy.fallback}")
    print(f"语义查询: {result.strategy.semantic_query[:80]}")
    print()

    print("Top 5 结果:")
    for i, a in enumerate(result.articles[:5], 1):
        print(f"  {i}. [sem={a.semantic_score:.3f} final={a.final_score:.3f}] PMID={a.pmid}")
        print(f"     {a.title[:60]}...")

    print()
    if result.quality:
        q = result.quality
        print("质量评估:")
        print(f"  等级: {q.get('assessment')}")
        print(f"  平均分: {q.get('avg_score')}")
        print(f"  区分度: {q.get('discrimination')}")
        print(f"  摘要覆盖率: {q.get('abstract_coverage')}")
        if q.get("suggestions"):
            print(f"  建议: {q['suggestions']}")

    print()
    print("=== 端到端验证通过 ===")


if __name__ == "__main__":
    main()
