from typing import List
from vllm import LLM, SamplingParams


import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler
import time
import wandb

class JSONDataset(Dataset):
    """
    从 JSON 文件中读取数据，每条数据至少包含一个 "context" 字段。
    可根据实际需求调整。
    """
    def __init__(self, json_path, max_samples=None):
        super().__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"system": item["system"], "conversations": item["conversations"]}

def collate_fn(batch):
    # batch 形如: [{"context": "..."}, {"context": "..."}...]
    contexts = [item["system"] + item['conversations'][0]['value'] for item in batch]
    return contexts

from torch.utils.data import Sampler

class SortedSampler(Sampler):
    """
    按照指定的字段或函数对整个数据集排序，然后依次产出索引。
    当达到 max_samples (若不为 None) 时，则停止产出。
    """

    def __init__(self, 
                 data_source: JSONDataset, 
                 sort_key=None, 
                 reverse=False, 
                 max_samples=None):
        """
        :param data_source: 传入自定义的 JSONDataset
        :param sort_key: 一个函数, 接受 data_source[i] (单条数据), 返回可比较的值，用于排序。
                         默认按 'context' 的长度排序。
        :param reverse:   排序方向，False 表示从小到大，True 表示从大到小
        :param max_samples: 若不为 None, 表示采样器最多产出多少个样本，超过则停止。
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.sort_key = sort_key if sort_key is not None else lambda x: len(x["conversations"][1]["value"])
        self.reverse = reverse
        self.max_samples = max_samples

        # 先把数据集的所有索引和对应的 key 值收集起来
        indexed_data = [(idx, self.sort_key(data_source[idx])) for idx in range(len(data_source))]
        # 按 key 排序
        indexed_data.sort(key=lambda x: x[1], reverse=self.reverse)
        # 得到排序后的索引列表
        self.sorted_indices = [x[0] for x in indexed_data]

    def __iter__(self):
        # 如果没有 max_samples，就遍历全部
        # 如果有 max_samples，就取前 max_samples 条
        stop = self.max_samples if self.max_samples is not None else len(self.sorted_indices)
        stop = min(stop, len(self.sorted_indices))  # 防止越界

        for i in range(stop):
            yield self.sorted_indices[i]

    def __len__(self):
        # 返回实际会产出的样本数
        if self.max_samples is not None:
            return min(len(self.data_source), self.max_samples)
        else:
            return len(self.data_source)

def multistep_rollout_batch_example(
    llm: str,
    contexts: List[str],
    sampling_params: SamplingParams,
    num_steps: int = 5,
    batch_idx: int = 0,
):
    """
    使用指定的模型，对 contexts 中的每个初始上下文做多步（multi-step）批量生成。
    每一轮生成后，如果检测到生成已完成(is_done)，则不再对该上下文做后续的推理/生成。
    
    :param model_path:  本地模型路径或在线模型名称（如 "gpt2", "facebook/opt-1.3b" 等）
    :param contexts:    存放多个初始上下文的列表，每个字符串相当于一个独立的对话/场景
    :param num_steps:   需要连续生成多少步
    :param max_tokens_per_step: 每步生成时最多生成的 token 数
    """
    
    step_start_time = time.time()  # 记录开始时间
    prompt_lengths = [len(ctx) for ctx in contexts]
    lengths = [0] * len(contexts)
    steps = [0] * len(contexts)

    print("-" * 40)

    # 用于记录每个上下文是否已完成生成，不再继续下一步
    is_done = [False] * len(contexts)

    for step in range(num_steps):
        # 只对 is_done == False 的上下文做生成
        active_contexts = [ctx for ctx, done in zip(contexts, is_done) if not done]
        
        # 如果全部完成，则提前退出
        if not active_contexts:
            print(f"在第 {step} 步时全部结束，提前退出循环。")
            break

        # 调用模型进行生成
        outputs = llm.generate(active_contexts, sampling_params)

        # 由于 outputs 的顺序与 active_contexts 对应，需要把它们映射回原 contexts 的顺序
        output_idx = 0  # 用于遍历 outputs
        print(f"=== 第 {step+1} 步生成结果 ===")

        for i in range(len(contexts)):
            # 跳过已完成的
            if is_done[i]:
                continue

            # 取对应结果
            result = outputs[output_idx]
            output_idx += 1

            # （若 n>1，可在 result.outputs 中选择最合适的候选）
            generated_text = result.outputs[0].text
            lengths[i] += len(generated_text)
            # print(f"- 上下文 {i+1} 的新增内容:\n{generated_text}\n")

            # 更新上下文
            contexts[i] += generated_text
            
            # 判断是否生成结束
            if result.outputs[0].finish_reason == "stop":
                steps[i] = step + 1
                is_done[i] = True
                # print(f"  -> 上下文 {i+1} 生成已满足结束条件，后续将不再生成。")


    step_end_time = time.time()  # 记录结束时间

    # 计算该次迭代的用时
    iteration_time = step_end_time - step_start_time
    try:
        avg_length = sum(lengths) / len(lengths)
    except:
        print(lengths)
        raise Exception("lengths is empty")
    
    try:
        var_length = torch.var(torch.tensor(lengths).float())
        std_length = torch.std(torch.tensor(lengths).float())
    except:
        print(lengths)
        raise Exception("lengths is empty")
    
    wandb.log({
                "iteration_time": iteration_time,
                "avg_steps": sum(steps) / len(steps),
                "prompt_len": sum(prompt_lengths)/ len(prompt_lengths),
                "avg_len": avg_length,
                "std_len": std_length,
                "batch_idx": batch_idx,
            },
            step=batch_idx)
    # print("=== 最终拼接后的上下文列表 ===")
    # for idx, ctx in enumerate(contexts):
    #     print(f"  上下文 {idx+1} (is_done={is_done[idx]}):\n{ctx}\n")


def main():
    dataset = JSONDataset("/mnt/hdfs/zhusiqi/modelhub/datasets--NovaSky-AI--Sky-T1_data_17k/Sky-T1_data_17k.json", max_samples=2048)

    # 构造一个随机采样器
    # random_sampler = SequentialSampler(dataset)
    sorted_sampler = SortedSampler(
        data_source=dataset,
        sort_key=lambda x: len(x["conversations"][1]['value']),  # 以 context 的长度排序
        reverse=False,                         # 从短到长
        max_samples=8192                # 如果不想限制总数，可以传 None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        sampler=sorted_sampler,
        collate_fn=collate_fn
    )

    model_path = "/mnt/hdfs/zhusiqi/modelhub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B" # 或换成你的本地大模型路径
    num_steps = 1
    max_tokens_per_step = 16384

    # 在你的脚本开始处初始化 wandb（注意项目名、实体名等可能需要替换）
    import wandb
    wandb.init(project="my-vllm-project", name="tp=2", config={
        "model": "deepseek-r1-7b",
        "batch_size": 64,
        "num_steps": num_steps,
        "max_tokens_per_step": max_tokens_per_step,
    })
    # 1. 初始化 LLM
    llm = LLM(model=model_path,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            tensor_parallel_size=2,
            disable_custom_all_reduce=True,)

    # 2. 设置采样参数（可以根据需要修改温度、top_p、top_k 等）
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens_per_step,
        # stop=["\nUser:"]  # 可根据实际需求添加 stop tokens
    )

    for batch_idx, contexts in enumerate(dataloader):
        # 3. 多步批量生成
        multistep_rollout_batch_example(
            llm=llm,
            contexts=contexts,
            sampling_params=sampling_params,
            num_steps=num_steps,
            batch_idx=batch_idx,
        )

if __name__ == "__main__":
    main()
