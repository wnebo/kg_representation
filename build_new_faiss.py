import faiss

index1 = faiss.read_index("index1.faiss")
index2 = faiss.read_index("index2.faiss")


xb1 = faiss.vector_to_array(index1.xb).reshape(index1.ntotal, index1.d)
xb2 = faiss.vector_to_array(index2.xb).reshape(index2.ntotal, index2.d)

# 确保 ID 是一样的
ids1 = faiss.vector_to_array(index1.id_map.ids) if isinstance(index1, faiss.IndexIDMap) else None
ids2 = faiss.vector_to_array(index2.id_map.ids) if isinstance(index2, faiss.IndexIDMap) else None

assert (ids1 == ids2).all(), "两个索引的 ID 不匹配"



alpha = 0.5  # 或者任何你想用的比例
beta = 0.5

import numpy as np

xb3 = alpha * xb1 + beta * xb2


d = xb3.shape[1]  # 维度
index3 = faiss.IndexFlatL2(d)  # 可改为其他类型

index3 = faiss.IndexIDMap(index3)
index3.add_with_ids(xb3.astype('float32'), ids1)


faiss.write_index(index3, "index3.faiss")



import faiss
import numpy as np

def combine_faiss_indexes(index_path1, index_path2, alpha=0.5, beta=0.5, output_path=None):
    # 读取两个索引
    index1 = faiss.read_index(index_path1)
    index2 = faiss.read_index(index_path2)
    
    # 检查维度一致性
    assert index1.d == index2.d, "索引维度不一致"
    assert index1.ntotal == index2.ntotal, "向量数量不一致"

    # 提取原始向量
    xb1 = faiss.vector_to_array(index1.xb).reshape(index1.ntotal, index1.d)
    xb2 = faiss.vector_to_array(index2.xb).reshape(index2.ntotal, index2.d)

    # 如果使用了 ID Map，提取 ID
    ids1 = ids2 = None
    if isinstance(index1, faiss.IndexIDMap):
        ids1 = faiss.vector_to_array(index1.id_map.ids)
    if isinstance(index2, faiss.IndexIDMap):
        ids2 = faiss.vector_to_array(index2.id_map.ids)
        assert (ids1 == ids2).all(), "两个索引的 ID 不一致"
    
    # 线性组合
    xb3 = alpha * xb1 + beta * xb2

    # 构建新索引
    index3 = faiss.IndexFlatL2(index1.d)
    if ids1 is not None:
        index3 = faiss.IndexIDMap(index3)
        index3.add_with_ids(xb3.astype('float32'), ids1)
    else:
        index3.add(xb3.astype('float32'))

    # 保存索引（可选）
    if output_path:
        faiss.write_index(index3, output_path)

    return index3
index3 = combine_faiss_indexes("index1.faiss", "index2.faiss", alpha=0.7, beta=0.3, output_path="index3.faiss")
