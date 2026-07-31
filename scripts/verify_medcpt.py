"""MedCPT 可行性验证：用 .nbib 案例数据测试语义编码与相似度计算。"""
import os
import sys

# 模型缓存设到 D 盘，避免 C 盘生成文件
os.environ["HF_HOME"] = "d:/trae/项目16：接单/a6_医学PubMed检索/streamlit/data/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "d:/trae/项目16：接单/a6_医学PubMed检索/streamlit/data/hf_cache"
# 使用国内镜像解决 HuggingFace 连接超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("=== MedCPT 可行性验证 ===\n")

# 1. 加载 MedCPT 双编码器
print("[1/4] 加载 MedCPT-Query-Encoder...")
from sentence_transformers import SentenceTransformer
import numpy as np
import time

t0 = time.time()
query_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder", device="cpu")
print(f"  加载耗时: {time.time()-t0:.1f}s")

print("[2/4] 加载 MedCPT-Article-Encoder...")
t0 = time.time()
article_encoder = SentenceTransformer("ncbi/MedCPT-Article-Encoder", device="cpu")
print(f"  加载耗时: {time.time()-t0:.1f}s")

# 2. 用 .nbib 案例的真实查询和摘要做测试
# 查询：来自项目需求 "化疗后中性粒细胞减少的血液学预测因子"
# 相关文献：PMID 39224807 的摘要（.nbib 案例文件中的第一篇）
# 无关文献：一段关于眼科的摘要

query = "hematological predictors for chemotherapy-induced febrile neutropenia"

relevant_abstract = (
    "OBJECTIVE: The aim of this study was to compare hematological parameters "
    "pre- and early post-chemotherapy, and evaluate their values for predicting "
    "febrile neutropenia (FN). METHODS: Patients diagnosed with malignant solid "
    "tumors receiving chemotherapy were included. Blood cell counts peri-chemotherapy "
    "and clinical information were retrieved from the hospital information system. "
    "We used the least absolute shrinkage and selection operator (LASSO) method for "
    "variable selection and fitted selected variables to a logistic model. RESULTS: "
    "Among hematological parameters, lower post-chemotherapy lymphocyte count and "
    "change percentage of platelet predicted an increased risk of FN. CONCLUSION: "
    "Peri-chemotherapy hematological markers improve the prediction of FN."
)

irrelevant_abstract = (
    "This study investigated the effects of blue light exposure on retinal pigment "
    "epithelium cells in age-related macular degeneration. Results showed that blue "
    "light induced oxidative stress and apoptosis in ARPE-19 cells, which could be "
    "attenuated by pretreatment with lutein and zeaxanthin."
)

# 3. 编码并计算相似度
print("\n[3/4] 编码查询与文献...")
t0 = time.time()
query_vec = query_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
print(f"  查询编码: {time.time()-t0:.3f}s, 维度={query_vec.shape}")

t0 = time.time()
article_vecs = article_encoder.encode(
    [relevant_abstract, irrelevant_abstract], convert_to_numpy=True, normalize_embeddings=True
)
print(f"  文献编码(2篇): {time.time()-t0:.3f}s, 维度={article_vecs.shape}")

# 4. 计算 cosine 相似度（已归一化，点积即 cosine）
print("\n[4/4] 语义相似度结果:")
sim_relevant = float(np.dot(query_vec[0], article_vecs[0]))
sim_irrelevant = float(np.dot(query_vec[0], article_vecs[1]))

print(f"  查询 vs 相关文献(化疗中性粒细胞减少): {sim_relevant:.4f}")
print(f"  查询 vs 无关文献(眼科黄斑变性):       {sim_irrelevant:.4f}")
print(f"  区分度(相关 - 无关):                   {sim_relevant - sim_irrelevant:.4f}")

if sim_relevant > sim_irrelevant:
    print("\n✓ 验证通过: MedCPT 能正确区分相关与无关文献")
    print(f"  相关文献相似度高出 {((sim_relevant/sim_irrelevant)-1)*100:.1f}%")
else:
    print("\n✗ 验证失败: 语义区分度不足")

print(f"\n结论: MedCPT 双编码器在 CPU 上可正常加载和推理")
print(f"  - 查询编码: ~{query_vec.shape[1]}d 向量")
print(f"  - 单篇编码速度可接受（批量2篇 {time.time()-t0:.2f}s 含输出）")
