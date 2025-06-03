import os
import pickle
import hashlib
import tiktoken

# ========== 工具函数 ==========
def compute_mdhash_id(content, prefix="chunk-"):
    md5 = hashlib.md5(content.encode("utf-8")).hexdigest()
    return f"{prefix}{md5}"

def chunking_by_token_size(
    tokens_list,
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        decoded_chunks = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(decoded_chunks):
            results.append({
                "tokens": lengths[i],
                "content": chunk.strip(),
                "chunk_order_index": i,
                "full_doc_id": doc_keys[index],
            })

    return results

# ========== 主函数 ==========
def preprocess_text_file(input_txt_path, output_pkl_path):
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    # 文档字典（只有一个文档）
    doc_id = compute_mdhash_id(raw_text, prefix="doc-")
    new_docs = {doc_id: {"content": raw_text}}

    # Tokenize
    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch([raw_text])
    doc_keys = [doc_id]

    # Chunking
    chunks = chunking_by_token_size(
        tokens_list=tokens,
        doc_keys=doc_keys,
        tiktoken_model=ENCODER,
        overlap_token_size=128,
        max_token_size=1024,
    )

    # 添加 chunk 的哈希 ID
    inserting_chunks = {
        compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk
        for chunk in chunks
    }

    # Save
    with open(output_pkl_path, "wb") as f:
        pickle.dump(inserting_chunks, f)

    print(f"✅ Saved {len(inserting_chunks)} chunks to {output_pkl_path}")


# ========== 调用 ==========
if __name__ == "__main__":
    input_txt = "data/source_text.txt"
    output_pkl = "data/processed_chunks.pkl"
    preprocess_text_file(input_txt, output_pkl)
