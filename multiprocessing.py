from concurrent.futures import ThreadPoolExecutor, as_completed

def worker(batchsize, index):
    print(f"[{threading.current_thread().name}] start idx={index}")
    # 模拟 I/O
    time.sleep(1)
    return (index, batchsize * index)

if __name__ == "__main__":
    import threading, time

    batchsize = 10
    idx_list = list(range(1, 9))  # idx 1–8

    results = []
    # 创建最多 4 个线程的池；不指定 max_workers 则默认是 CPU 核心数*5
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务
        futures = [executor.submit(worker, batchsize, idx) for idx in idx_list]
        # as_completed：一有完成就处理
        for fut in as_completed(futures):
            idx, res = fut.result()
            print(f"[Main] Got result for idx={idx}: {res}")
            results.append((idx, res))

    print("所有结果：", sorted(results))
