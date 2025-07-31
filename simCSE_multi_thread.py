import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import re
from typing import List, Dict, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 配置
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
PAGENUM2CONTEXT_PATH = 'pagenum2context.pkl'

# 文件路径
QUESTIONS2PAGE_PATH = 'questions2page.pkl'
PAGE2QUESTIONS_PATH = 'page2questions.pkl'
FAISS_INDEX_PATH = 'question2vec_faiss.index'
QUESTION_LIST_PATH = 'question_list.pkl'  # 存储问题列表，用于索引映射

def load_book_data(path: str) -> Dict:
    """加载书籍数据"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_questions_for_page(page_num: int, content: str, llm_call_func) -> List[str]:
    """
    为单页内容生成相关问题
    """
    prompt = f"""
基于以下文本内容，生成3-5个相关的问题。这些问题应该能够通过该文本内容来回答。
请直接返回问题，每行一个，不要添加编号或其他格式。

文本内容：
{content}

生成的问题：
"""
    
    try:
        response = llm_call_func(prompt)
        
        # 解析问题
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        
        # 清理问题格式
        cleaned_questions = []
        for q in questions:
            # 移除可能的编号
            q = re.sub(r'^\d+\.\s*', '', q)
            q = re.sub(r'^[•\-\*]\s*', '', q)
            if q and not q.startswith('问题') and len(q) > 5:
                cleaned_questions.append(q)
        
        return cleaned_questions[:5]
        
    except Exception as e:
        print(f"为页码 {page_num} 生成问题时出错: {e}")
        return []

def process_page_worker(page_data: Tuple[int, str], llm_call_func) -> Tuple[int, List[str]]:
    """
    单个页面处理的工作函数
    """
    page_num, content = page_data
    questions = generate_questions_for_page(page_num, content, llm_call_func)
    return page_num, questions

def build_question_mappings_threaded(pagenum2context: Dict, llm_call_func, max_workers: int = 5):
    """
    使用多线程构建问题映射和Faiss索引
    
    Args:
        pagenum2context: 页码到内容的字典
        llm_call_func: LLM调用函数
        max_workers: 最大线程数
    """
    print(f"开始构建问题映射（使用 {max_workers} 个线程）...")
    start_time = time.time()
    
    # 初始化embedding模型
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 存储结构
    questions2page = {}      # 问题 -> 页码
    page2questions = {}      # 页码 -> 问题列表
    all_questions = []       # 所有问题的列表（用于Faiss索引映射）
    
    # 准备页面数据
    page_items = list(pagenum2context.items())
    total_pages = len(page_items)
    processed_pages = 0
    
    print(f"开始处理 {total_pages} 页内容...")
    
    # 使用线程池处理页面
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_page = {
            executor.submit(process_page_worker, page_data, llm_call_func): page_data[0] 
            for page_data in page_items
        }
        
        # 收集结果
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                page_num, questions = future.result()
                processed_pages += 1
                
                if questions:
                    # 存储映射关系
                    page2questions[page_num] = questions
                    
                    for question in questions:
                        questions2page[question] = page_num
                        all_questions.append(question)
                    
                    print(f"✓ 页码 {page_num}: 生成 {len(questions)} 个问题 [{processed_pages}/{total_pages}]")
                else:
                    print(f"✗ 页码 {page_num}: 未生成问题 [{processed_pages}/{total_pages}]")
                    
            except Exception as e:
                print(f"✗ 页码 {page_num} 处理失败: {e} [{processed_pages}/{total_pages}]")
                processed_pages += 1
    
    processing_time = time.time() - start_time
    print(f"\n问题生成完成！耗时: {processing_time:.2f}秒")
    print(f"共生成 {len(all_questions)} 个问题，开始计算embeddings...")
    
    # 批量生成embeddings
    if all_questions:
        embeddings = embedding_model.encode(all_questions, show_progress_bar=True)
        
        # 构建Faiss索引
        print("构建Faiss索引...")
        dimension = embeddings.shape[1]
        
        # 使用内积相似度索引
        index = faiss.IndexFlatIP(dimension)
        
        # 归一化embedding向量（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        
        # 添加到索引
        index.add(embeddings.astype('float32'))
        
        print(f"Faiss索引构建完成，维度: {dimension}, 向量数量: {index.ntotal}")
    else:
        print("没有生成任何问题，跳过索引构建")
        return
    
    # 保存所有数据
    print("保存数据...")
    
    # 保存映射关系
    with open(QUESTIONS2PAGE_PATH, 'wb') as f:
        pickle.dump(questions2page, f)
    
    with open(PAGE2QUESTIONS_PATH, 'wb') as f:
        pickle.dump(page2questions, f)
    
    # 保存问题列表（用于索引映射）
    with open(QUESTION_LIST_PATH, 'wb') as f:
        pickle.dump(all_questions, f)
    
    # 保存Faiss索引
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    total_time = time.time() - start_time
    print(f"所有数据保存完成！总耗时: {total_time:.2f}秒")
    
    # 输出统计信息
    stats = {
        'total_pages': len(pagenum2context),
        'pages_processed': processed_pages,
        'total_questions': len(all_questions),
        'avg_questions_per_page': len(all_questions) / len(pagenum2context) if pagenum2context else 0,
        'pages_with_questions': len(page2questions),
        'processing_time': f"{processing_time:.2f}s",
        'total_time': f"{total_time:.2f}s"
    }
    print("统计信息:", stats)

# 保留原来的单线程版本（兼容性）
def build_question_mappings(pagenum2context: Dict, llm_call_func):
    """
    单线程版本（兼容性保留）
    """
    return build_question_mappings_threaded(pagenum2context, llm_call_func, max_workers=1)

def load_mappings_and_index():
    """
    加载所有映射数据和Faiss索引
    """
    try:
        # 加载映射关系
        with open(QUESTIONS2PAGE_PATH, 'rb') as f:
            questions2page = pickle.load(f)
        
        with open(PAGE2QUESTIONS_PATH, 'rb') as f:
            page2questions = pickle.load(f)
            
        with open(QUESTION_LIST_PATH, 'rb') as f:
            question_list = pickle.load(f)
        
        # 加载Faiss索引
        index = faiss.read_index(FAISS_INDEX_PATH)
        
        print(f"数据加载成功！问题数量: {len(question_list)}, 索引大小: {index.ntotal}")
        
        return questions2page, page2questions, question_list, index
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请先运行构建函数")
        return None, None, None, None

def search_similar_questions(query: str, index, question_list, questions2page, top_k: int = 5):
    """
    使用Faiss搜索最相似的问题
    """
    # 初始化embedding模型
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # 生成查询的embedding
    query_embedding = embedding_model.encode([query])
    
    # 归一化（用于余弦相似度）
    faiss.normalize_L2(query_embedding)
    
    # 在Faiss索引中搜索
    similarities, indices = index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < len(question_list):  # 确保索引有效
            question = question_list[idx]
            page_num = questions2page.get(question, None)
            
            results.append({
                'question': question,
                'similarity': float(similarity),
                'page_num': page_num,
                'rank': i + 1
            })
    
    return results

def search_pages(query: str, pagenum2context: Dict, top_k: int = 3):
    """
    根据查询返回最相关的页面内容
    """
    # 加载数据
    questions2page, page2questions, question_list, index = load_mappings_and_index()
    
    if index is None:
        print("请先构建索引")
        return []
    
    # 搜索相似问题
    similar_questions = search_similar_questions(query, index, question_list, questions2page, top_k * 2)
    
    # 去重页面并组织结果
    results = []
    seen_pages = set()
    
    for item in similar_questions:
        page_num = item['page_num']
        if page_num and page_num not in seen_pages and len(results) < top_k:
            seen_pages.add(page_num)
            
            result = {
                'page_num': page_num,
                'similar_question': item['question'],
                'similarity': item['similarity'],
                'content': pagenum2context.get(page_num, ""),
                'rank': len(results) + 1
            }
            results.append(result)
    
    return results

# 使用示例函数
def build_index_example(llm_call_func, max_workers: int = 5):
    """
    构建索引的示例（多线程版本）
    
    Args:
        llm_call_func: LLM调用函数
        max_workers: 最大线程数，建议根据你的LLM API限制设置
    """
    # 加载书籍数据
    pagenum2context = load_book_data(PAGENUM2CONTEXT_PATH)
    print(f"加载了 {len(pagenum2context)} 页内容")
    
    # 构建问题映射和索引（多线程）
    build_question_mappings_threaded(pagenum2context, llm_call_func, max_workers=max_workers)

def build_index_example_single_thread(llm_call_func):
    """
    构建索引的示例（单线程版本）
    """
    # 加载书籍数据
    pagenum2context = load_book_data(PAGENUM2CONTEXT_PATH)
    print(f"加载了 {len(pagenum2context)} 页内容")
    
    # 构建问题映射和索引（单线程）
    build_question_mappings(pagenum2context, llm_call_func)

def search_example():
    """
    搜索的示例
    """
    # 加载书籍数据
    pagenum2context = load_book_data(PAGENUM2CONTEXT_PATH)
    
    # 执行搜索
    query = "如何学习Python编程？"
    results = search_pages(query, pagenum2context, top_k=3)
    
    print(f"\n查询: {query}")
    print("=" * 50)
    
    for result in results:
        print(f"排名: {result['rank']}")
        print(f"页码: {result['page_num']}")
        print(f"相似问题: {result['similar_question']}")
        print(f"相似度: {result['similarity']:.3f}")
        print(f"内容预览: {result['content'][:100]}...")
        print("-" * 50)

def check_index_status():
    """
    检查索引状态
    """
    try:
        questions2page, page2questions, question_list, index = load_mappings_and_index()
        
        if index:
            print(f"索引状态: 正常")
            print(f"问题总数: {len(question_list)}")
            print(f"页面数量: {len(page2questions)}")
            print(f"平均每页问题数: {len(question_list) / len(page2questions):.2f}")
            
            # 显示一些示例问题
            print("\n示例问题:")
            for i, question in enumerate(question_list[:5]):
                page_num = questions2page.get(question, "未知")
                print(f"{i+1}. {question} (页码: {page_num})")
        else:
            print("索引未构建或加载失败")
            
    except Exception as e:
        print(f"检查索引时出错: {e}")

# 主要使用流程
if __name__ == "__main__":
    # 假设这是你的LLM调用函数
    def my_llm_call(prompt):
        # 替换为你的实际LLM调用代码
        # return your_llm_function(prompt)
        pass
    
    print("RAG问题映射系统（多线程版本）")
    print("1. 首次使用，构建索引（多线程，推荐）:")
    print("   build_index_example(my_llm_call, max_workers=5)")
    print("\n2. 首次使用，构建索引（单线程）:")
    print("   build_index_example_single_thread(my_llm_call)")
    print("\n3. 日常搜索:")
    print("   search_example()")
    print("\n4. 检查索引状态:")
    print("   check_index_status()")
    
    print("\n线程数建议:")
    print("- 如果你的LLM API有速率限制（如每分钟调用次数），建议设置较小的max_workers（2-5）")
    print("- 如果没有严格限制，可以设置更大的值（5-10）来加速处理")
    print("- 过多线程可能导致API限制或网络拥塞，建议逐步测试最佳值")
    
    # 取消注释来运行
    # build_index_example(my_llm_call, max_workers=5)  # 多线程构建
    # build_index_example_single_thread(my_llm_call)   # 单线程构建
    # search_example()  # 搜索测试
    # check_index_status()  # 检查状态
