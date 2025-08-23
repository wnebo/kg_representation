##############################################
一步验证公司服务是否可用（能力探针）
##############################################
from openai import OpenAI

client = OpenAI(
    base_url="http://your-company-llm-endpoint/v1",
    api_key="",  # 无鉴权可空
)

resp = client.chat.completions.create(
    model="llama3.3-70b",  # 你公司提供的名字
    messages=[{"role": "user", "content": "1+1 等于几？只回答数字"}],
    max_tokens=16,
    logprobs=True,
    top_logprobs=5,   # 必须≥2
    n=2,              # 故意要多条，方便测试
    extra_body={      # 测试是否能透传 vllm_xargs（不支持也不会报错就说明被吃掉了）
        "vllm_xargs": {
            "enable_conf": True,
            "window_size": 64,
            "threshold": 10.0
        }
    }
)

print(hasattr(resp.choices[0], "logprobs"), resp.choices[0].logprobs is not None)
# True/False 查看是否真的返回了logprobs


##############################################
方案 A：公司 LLM（OpenAI 兼容）— Online DeepConf 最小可跑脚本
##############################################
from openai import OpenAI
import numpy as np

client = OpenAI(base_url="http://your-company-llm-endpoint/v1", api_key="")

MODEL = "llama3.3-70b"  # 公司模型名
PROMPT = "请一步步推理：23 × 42 的结果？最后用 \\boxed{} 给出答案。"
WARMUP_TRACES = 8
TOTAL_TRACES  = 16   # 总预算 = 预热 + 在线
WINDOW_SIZE   = 128  # 置信度滑窗（越大越稳）
PERCENTILE    = 90   # 阈值选用分位数

def min_sliding_mean_conf(choice):
    # 基于 top_logprobs 计算每步的“候选均值负对数概率”（= 置信度）
    confs = []
    for t in (choice.logprobs.content or []):
        alts = t.top_logprobs or []
        if len(alts) >= 1:
            mean_alt = -sum(a.logprob for a in alts)/len(alts)
        else:
            mean_alt = 0.0
        confs.append(mean_alt)
    if not confs:
        return 0.0
    # 滑窗均值的最小值
    if len(confs) <= WINDOW_SIZE:
        return float(np.mean(confs))
    # 快速滑窗均值
    arr = np.array(confs, dtype=float)
    cs = np.cumsum(arr)
    win = cs[WINDOW_SIZE-1:] - np.concatenate(([0], cs[:-WINDOW_SIZE]))
    win_mean = win / WINDOW_SIZE
    return float(win_mean.min())

# 1) 预热：拿阈值
warm = client.chat.completions.create(
    model=MODEL,
    messages=[{"role":"user","content":PROMPT}],
    n=WARMUP_TRACES,
    max_tokens=512,
    temperature=0.8,
    top_p=0.95,
    logprobs=True,
    top_logprobs=20,
    extra_body={"top_k": 0}
)

mins = [min_sliding_mean_conf(c) for c in warm.choices]
bar = float(np.percentile(mins, PERCENTILE))
print("Warmup mins:", mins, "→ bar =", round(bar,3))

# 2) 在线：带 server-side 早停（DeepConf）
real_n = TOTAL_TRACES - WARMUP_TRACES
final = client.chat.completions.create(
    model=MODEL,
    messages=[{"role":"user","content":PROMPT}],
    n=real_n,
    max_tokens=2048,
    temperature=0.8,
    top_p=0.95,
    logprobs=True,
    top_logprobs=20,
    extra_body={
        "top_k": 0,
        "vllm_xargs": {
            "enable_conf": True,
            "window_size": WINDOW_SIZE,
            "threshold": bar
        }
    }
)

# 3) 投票（多数或加权都可；这里用多数）
def extract_boxed(txt):
    if "boxed" not in txt: return None
    part = txt.split("boxed",1)[1]
    if part.startswith("{"):
        depth, ans = 1, []
        for ch in part[1:]:
            if ch=="{": depth+=1
            elif ch=="}":
                depth-=1
                if depth==0: break
            ans.append(ch)
        return "".join(ans).strip()
    return part.split("$")[0].strip()

answers = []
for c in warm.choices + final.choices:
    a = extract_boxed(c.message.content or "")
    if a: answers.append(a)

# 多数表决
from collections import Counter
vote = Counter(answers).most_common(1)[0][0] if answers else None
print("VOTE:", vote)


##############################################
方案 A：公司 LLM（OpenAI 兼容）— Online DeepConf 最小可跑脚本 
（如果公司没集成 DeepConf → 你请求里加 enable_conf 也不会有作用

你可以做的只是客户端早停（模拟策略）：

通过 stream=True 拿到 token 流

本地计算置信度并在必要时中止连接

节省剩余未生成的 token，但 推理到被你中断的那一步的 token 依然会计费或消耗算力）
##############################################
from openai import OpenAI
import numpy as np
import threading

client = OpenAI(base_url="http://your-company-llm-endpoint/v1", api_key="")

MODEL = "llama3.3-70b"   # 你的公司模型名
PROMPT = "请一步步推理：23 × 42 的结果？最后用 \\boxed{} 给出答案。"

WINDOW_SIZE = 128
CONF_BAR = 10.0          # 先随便设个阈值；更严谨的做法是先用若干 warmup trace 求分位数作为阈值
N_TRACES = 8             # 多条 trace → 每条用一个独立流，便于单独早停

def should_stop(confs, window=WINDOW_SIZE, bar=CONF_BAR):
    if not confs:
        return False
    if len(confs) < window:
        avg = float(np.mean(confs))
        return avg < bar
    arr = np.array(confs, dtype=float)
    cs = np.cumsum(arr)
    win = cs[window-1:] - np.concatenate(([0], cs[:-window]))
    min_mean = float((win / window).min())
    return min_mean < bar

def run_one_trace(idx):
    conf_seq = []
    text = []

    with client.chat.completions.stream(
        model=MODEL,
        messages=[{"role":"user","content":PROMPT}],
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
        logprobs=True,
        top_logprobs=20,          # 必须≥2，越大越稳
        extra_body={"top_k": 0},  # 可选
    ) as stream:
        for event in stream:
            if event.type == "token":  # SDK 不同版本事件名可能略有差异：有的叫 'delta'，请按你环境调整
                t = event.token
                text.append(t.value)

                # 计算该步置信度（候选的平均 -logprob）
                tlps = t.logprob.top_logprobs if hasattr(t, "logprob") and t.logprob else []
                if tlps:
                    mean_alt_nll = -sum(x.logprob for x in tlps) / len(tlps)
                else:
                    mean_alt_nll = 0.0
                conf_seq.append(mean_alt_nll)

                # 在线检查是否需要早停
                if should_stop(conf_seq):
                    stream.close()  # 立刻终止该 trace 的后续生成
                    print(f"[trace {idx}] early-stopped, tokens={len(text)}")
                    break

            elif event.type == "error":
                print(f"[trace {idx}] error: {event}")
                break

    full_text = "".join(text)
    return full_text

# 并行跑多条 trace（每条独立流，便于单独早停）
threads, outputs = [], [None]*N_TRACES
for i in range(N_TRACES):
    th = threading.Thread(target=lambda k=i: outputs.__setitem__(k, run_one_trace(k)))
    th.start()
    threads.append(th)

for th in threads:
    th.join()

# 简单“多数表决”提取答案
def extract_boxed(txt):
    if "boxed" not in txt: return None
    part = txt.split("boxed",1)[1]
    if part.startswith("{"):
        depth, ans = 1, []
        for ch in part[1:]:
            if ch=="{": depth+=1
            elif ch=="}":
                depth-=1
                if depth==0: break
            ans.append(ch)
        return "".join(ans).strip()
    return part.split("$")[0].strip()

answers = [extract_boxed(x or "") for x in outputs if x]
from collections import Counter
vote = Counter([a for a in answers if a]).most_common(1)
print("VOTE:", vote[0][0] if vote else None)


##############################################
方案 B：本地自建 vLLM（推荐可控）
安装 vLLM 并应用 DeepConf 补丁/使用相应 PR 构建

###
# 1) 克隆 vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 2) 抓取 PR 分支（以 PR #23201 为例）
git fetch origin pull/23201/head:deepconf-pr
git checkout deepconf-pr
# 可选：切到作者标注测试过的 commit（例如你文档里提到的 31f09c6...）
# git checkout 31f09c615f4f067dba765ce5fe7d00d880212a6d

# 3) 构建安装（开发模式）
VLLM_USE_PRECOMPILED=1 pip install -e .
# 或者用 uv/pipx，任选其一
# VLLM_USE_PRECOMPILED=1 uv pip install --editable .
###

启动服务：

vllm serve Qwen/Qwen3-8B --port 8000 -tp 1 --gpu-memory-utilization 0.8


把上面 方案 A 的 client = OpenAI(base_url="http://localhost:8000/v1", api_key="")，MODEL="Qwen/Qwen3-8B" 改掉即可直接跑。
##############################################
