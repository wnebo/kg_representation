import json

def load_and_convert_kg_to_descriptions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_descriptions = {}
    for chunk_id, chunk in data.items():
        kg = chunk.get("kg", [])
        desc_list = []

        for item in kg:
            if "entity" in item:
                e = item["entity"]
                sentence = f"The entity '{e['name']}' is a {e['type']} described as: {e['description']}."
                desc_list.append(sentence)
            elif "relation" in item:
                r = item["relation"]
                sentence = (
                    f"There is a relation between '{r['entity1']}' and '{r['entity2']}': {r['description']} "
                    f"(strength: {r.get('strength', 'unknown')})."
                )
                desc_list.append(sentence)

        final_description = " ".join(desc_list)
        chunk_descriptions[chunk_id] = final_description

    return chunk_descriptions


import numpy as np
import faiss
import pickle

def build_kg_faiss_index(chunk_descriptions: dict, embedder, save_dir="data"):
    chunk_ids = list(chunk_descriptions.keys())
    descriptions = list(chunk_descriptions.values())

    # 假设 embedder 输出是 List[List[float]]
    embeddings = embedder(descriptions)
    embedding_matrix = np.array(embeddings).astype("float32")

    # 构建 FAISS 索引
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    # 存储索引 & id 映射
    faiss.write_index(index, f"{save_dir}/kg_faiss.index")
    with open(f"{save_dir}/chunk_id_map.pkl", "wb") as f:
        pickle.dump(chunk_ids, f)

    return index, chunk_ids


def search_similar_chunks(query_text, embedder, faiss_index_path, chunk_id_map_path, top_k=5):
    # 载入 index 和 id 映射
    index = faiss.read_index(faiss_index_path)
    with open(chunk_id_map_path, "rb") as f:
        chunk_ids = pickle.load(f)

    # 向量化 query
    query_vec = np.array([embedder([query_text])[0]]).astype("float32")

    # 查询
    D, I = index.search(query_vec, top_k)
    results = [chunk_ids[i] for i in I[0]]

    return results


import faiss
import numpy as np

# ---- 1. 准备示例数据 ----
# 假设有 5 个 chunk，每个向量维度为 4
dim = 4
num_chunks = 5

# 随机生成 embedding 矩阵（shape = [5,4]）
np.random.seed(123)
embedding_matrix = np.random.random((num_chunks, dim)).astype('float32')

# 你自己的 chunk_id 列表（必须是 int64）
chunk_ids = np.array([101, 205, 309, 412, 523], dtype='int64')

# ---- 2. 创建 FAISS 索引并 wrap 成 IndexIDMap2 ----
# 需要 faiss >= 1.6.3 才有 IndexIDMap2
flat = faiss.IndexFlatL2(dim)           # 底层用 L2 距离
index = faiss.IndexIDMap2(flat)         # 包一层 IDMap2

# ---- 3. 一起添加向量和自定义 ID ----
index.add_with_ids(embedding_matrix, chunk_ids)
print(f"Total vectors in index: {index.ntotal}")
# 输出：Total vectors in index: 5

# ---- 4. 做一次搜索 ----
# 这里用第一条向量自身作为查询，找出最近的 3 个 chunk
query = embedding_matrix[0:1]           # shape = (1,4)
top_k = 3

D, I = index.search(query, top_k)
# D: 距离矩阵，shape = (1,3)
# I: 返回的 chunk_id，shape = (1,3)
print("Distances:", D)
print("Returned chunk_ids:", I)
# 例如输出：
# Distances: [[0.       0.3573  1.0456 ]]
# Returned chunk_ids: [[101 205 309]]

# ---- 5. 可选：保存和加载 ----
faiss.write_index(index, "kg_faiss.index")
# 下次可以直接读回、继续增删或搜索
# loaded = faiss.read_index("kg_faiss.index")
