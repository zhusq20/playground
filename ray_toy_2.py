import ray
from vllm import LLM, SamplingParams
import time
import statistics
# 启动 Ray，本地运行
ray.init()  # 若在集群环境下，可配置地址等参数

# 定义 Ray Actor，用于封装 vLLM 引擎
class VLLMEngine:
    def __init__(self, model_name: str, tp_size: int):
        """初始化 vLLM 引擎，加载指定模型，设置tensor parallel大小"""
        # 创建LLM引擎，设置tensor_parallel_size
        self.llm = LLM(model=model_name, tensor_parallel_size=tp_size, disable_custom_all_reduce=True,)
                       
        # 可以在此打印日志确认加载完成
        print(f"Initialized vLLMEngine with model {model_name} (TP={tp_size})")

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 1.0):
        """对给定prompt执行推理，并返回生成的文本"""
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        output = self.llm.generate(prompt, params)  # 调用vLLM生成 [oai_citation_attribution:5‡docs.vllm.ai](https://docs.vllm.ai/en/latest/serving/distributed_serving.html#:~:text=To%20run%20multi,run%20inference%20on%204%20GPUs)
        # `output` 是包含生成结果的列表，这里返回首个结果文本
        return output[0].outputs[0].text  # 假设每个请求只生成一个输出文本

# 使用 Ray 的 .options 指定每个引擎actor使用的 GPU 数量，启动两个引擎
EngineActor = ray.remote(VLLMEngine)
# 创建tensor parallel大小为2的引擎（占用2个GPU）
EngineActor_tp2 = EngineActor.options(num_gpus=2)
engine_tp2 = EngineActor_tp2.remote("/mnt/hdfs/zhusiqi/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B", 2)
# 创建tensor parallel大小为4的引擎（占用4个GPU）
EngineActor_tp4 = EngineActor.options(num_gpus=4)
engine_tp4 = EngineActor_tp4.remote("/mnt/hdfs/zhusiqi/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B", 4)

# 调度器初始化：当前每个引擎正在处理的请求数
inflight_tp2 = 0
inflight_tp4 = 0


# 准备列表模拟一批待处理的推理请求（这里每个请求是一个prompt字符串）
requests = [
    "Hello, how are you?",
    "什么是张量并行？请用中文回答。",
    "Explain the theory of relativity in simple terms.",
    "给出中国的首都和人口。",
    "What are the main features of Qwen2.5-7B model?",
    # ... 可根据需要添加更多请求
]


# 定义一个函数，把 requests 分成若干批
def chunk_requests(req_list, batch_size):
    """Yield successive batch_size-sized chunks."""
    for i in range(0, len(req_list), batch_size):
        yield req_list[i:i+batch_size]

batch_size = 2  # 每个批2条请求
batches = list(chunk_requests(requests, batch_size))

# 用于记录每个请求启动的时间和对应的引擎，用来计算响应延迟
future_info = {}  # {future: (engine_id, start_time)}

# 列表用于记录两种配置的延迟
latencies_tp2 = []
latencies_tp4 = []

# 逐个发送请求
for batch in batches:
    # 根据当前负载选择引擎
    if inflight_tp2 <= inflight_tp4:
        # 分配给 TP=2 引擎
        start_time = time.time()
        future = engine_tp2.generate.remote(batch)
        inflight_tp2 += 1
        future_info[future] = ("TP2", start_time, batch)
    else:
        # 分配给 TP=4 引擎
        start_time = time.time()
        future = engine_tp4.generate.remote(batch)
        inflight_tp4 += 1
        future_info[future] = ("TP4", start_time, batch)

# ----------------- 2) 使用 ray.wait() 异步获取结果 -----------------
unfinished = list(future_info.keys())  # 把全部 future 放进“未完成”列表

while unfinished:
    # ray.wait 第一个参数是等待列表, num_returns=1 表示一次只拿到一个完成任务
    # timeout=None 表示一直等到至少有一个任务完成才返回
    done_futures, unfinished = ray.wait(unfinished, num_returns=1, timeout=None)

    # 处理已完成任务
    for future in done_futures:
        engine_id, start_time, prompt = future_info[future]
        end_time = time.time()
        latency = end_time - start_time

        # 获取推理结果（此时不会阻塞，因为这个 future 已完成）
        result_text = ray.get(future)

        # 记录延迟 & 更新 inflight
        if engine_id == "TP2":
            latencies_tp2.append(latency)
            inflight_tp2 -= 1
        else:
            latencies_tp4.append(latency)
            inflight_tp4 -= 1

        # 打印信息
        print(f"[{engine_id}] Prompt: {prompt[:20]}... -> "
              f"Result: {result_text[:30]}... || "
              f"Latency: {latency:.3f}s")

# ----------------- 3) 输出统计对比 -----------------
print("\n=== 性能对比: TP=2 vs TP=4 ===")
def safe_stat(lst):
    return (statistics.mean(lst) if lst else 0.0,
            statistics.median(lst) if lst else 0.0)

avg_tp2, med_tp2 = safe_stat(latencies_tp2)
avg_tp4, med_tp4 = safe_stat(latencies_tp4)

print(f"TP=2: 请求数={len(latencies_tp2)}, 平均延迟={avg_tp2:.3f}, 中位数={med_tp2:.3f}")
print(f"TP=4: 请求数={len(latencies_tp4)}, 平均延迟={avg_tp4:.3f}, 中位数={med_tp4:.3f}")

ray.shutdown()