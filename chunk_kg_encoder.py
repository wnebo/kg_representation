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
