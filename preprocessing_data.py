# 清理

from ftfy import fix_text

def clean_chunk(chunk):
    chunk = fix_text(chunk)
    chunk = re.sub(r'[\x00-\x1F\x7F]', '', chunk)
    return chunk.strip()

fixed_chunks = [clean_chunk(c) for c in all_chunks]


# 判断

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

def is_problematic_chunk(chunk):
    try:
        encoded = tokenizer(chunk, return_tensors="pt")
        decoded = tokenizer.decode(encoded["input_ids"][0])
        if decoded != chunk:
            print("⚠️ 编码/解码不一致")
            return True
        return False
    except Exception as e:
        print(f"❌ Tokenizer 报错: {e}")
        return True


for i, chunk in enumerate(all_chunks):
    if is_problematic_chunk(chunk):
        print(f"❗️Chunk #{i} 有问题")


# 模拟

def test_llm(chunk):
    try:
        response = llama_model.generate(chunk)  # 换成你的调用方法
        return True
    except Exception as e:
        print(f"⚠️ LLM 报错: {e}")
        return False
bad_chunks = [c for c in all_chunks if not test_llm(c)]



# 检索：元数据过滤
# 帮助将所有chunk放在一起 只不过增加 文件名字 公司名字 



# 融和text/kg
✅ 融合方案 1：加权融合 Top-k 检索
top_chunks_text = search_text_chunks(query_embedding)   # [(chunk_id, score), ...]
top_chunks_kg = search_kg_chunks(query_embedding)       # [(chunk_id, score), ...]

from collections import defaultdict

def fuse_scores(text_chunks, kg_chunks, alpha=0.6):
    scores = defaultdict(float)
    for cid, s in text_chunks:
        scores[cid] += alpha * s
    for cid, s in kg_chunks:
        scores[cid] += (1 - alpha) * s
    return sorted(scores.items(), key=lambda x: -x[1])

✅ 融合方案 2：补充策略（覆盖优化）
top_chunks_text = search_text_chunks(query_embedding)
covered_contexts = get_contexts(top_chunks_text)

if not is_context_covered(covered_contexts, ground_truth_context):
    top_chunks_kg = search_kg_chunks(query_embedding)
    # 选出与原始结果不重合的 KG chunks，补充进去
    for cid, _ in top_chunks_kg:
        if cid not in [x[0] for x in top_chunks_text]:
            top_chunks_text.append((cid, fallback_score))
            if len(top_chunks_text) >= desired_k:
                break


✅ 融合方案 3：使用对比学习增强 query-KG 相关性（SimCSE 思路）
构造训练集：

query 和相关 chunk（KG + 原文）作为正样本对

query 和无关 chunk 为负样本

训练一个双塔模型（query encoder + chunk encoder），让模型主动学习哪些 chunk（不管是 text 还是 KG）更 relevant。

优点：

更强表达能力

支持个性化 recall（query-aware）


✅ 融合方案 4：使用对比学习增强 query-KG 相关性（SimCSE 思路）

阶段 1：KG ↔ Chunk 原文 对比学习（结构 ↔ 语义 对齐）
正样本：同一个 chunk 的 KG 和原文

负样本：来自其他 chunk 的 KG 与原文交叉配对

目的：让模型学会将 KG 的结构信息和原文的语义表示对齐。

阶段 2：Query ↔ Chunk 表示 对比学习（需求 ↔ 内容 匹配）
正样本：query 和相关 chunk（KG + 原文融合后的表示）

负样本：query 和无关 chunk 的融合表示

目的：让模型学会识别哪个 chunk（融合信息）对 query 更 relevant。


# rerank 需要训练

# ground_truth_context--ground_truth_chunks

✅ 方案 1：预处理时保留每个 chunk 的“原文 span”
比如，在你分 chunk 的时候，加上：

python
复制
编辑
{
  "chunk_id": "...",
  "start_char": 1034,
  "end_char": 2050,
  ...
}
然后你可以直接判断某个 chunk 是否“落在” ground-truth context 所在位置（例如：overlap ≥ 50%），作为真值判断。

优点：

不依赖 encoder 表示

清晰客观

可用于准确评估 recall、precision

