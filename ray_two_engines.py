import ray
from ray.util.placement_group import placement_group

# 启动Ray
ray.init()

# 为 TP=2 和 TP=4 分别创建 Placement Group
pg_tp2 = placement_group([{"GPU": 2}], strategy="PACK")
pg_tp4 = placement_group([{"GPU": 4}], strategy="PACK")

# 等待资源就绪
ray.get(pg_tp2.ready())
ray.get(pg_tp4.ready())

@ray.remote(num_gpus=2)
class VLLMEngine:
    def __init__(self, model_name: str, tp_size: int):
        from vllm import LLM
        self.llm = LLM(model=model_name, tensor_parallel_size=tp_size)

    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts, sampling_params)
    
model_name = "meta-llama/Llama-2-7b-hf"

engine_tp2 = VLLMEngine.options(
    placement_group=pg_tp2,
    placement_group_bundle_index=0,
    placement_group_capture_child_tasks=True,
).remote(model_name, 2)

engine_tp4 = VLLMEngine.options(
    placement_group=pg_tp4,
    placement_group_bundle_index=0,
    placement_group_capture_child_tasks=True,
).remote(model_name, 4)


class InferenceRouter:
    def __init__(self, engine_tp2, engine_tp4):
        self.engine_tp2 = engine_tp2
        self.engine_tp4 = engine_tp4

    def route(self, prompt: str):
        # 简单分流规则：短的给 TP=2，长的给 TP=4
        if len(prompt) < 100:
            return self.engine_tp2
        else:
            return self.engine_tp4

    async def generate(self, prompt: str, sampling_params):
        engine = self.route(prompt)
        result = await engine.generate.remote([prompt], sampling_params)
        return ray.get(result)[0].outputs[0].text
    

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 64

router = None  # 之后在 main 中初始化

@app.post("/generate")
async def generate_text(req: PromptRequest):
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    output = await router.generate(req.prompt, sampling_params)
    return {"response": output}


if __name__ == "__main__":
    import asyncio

    model_name = "meta-llama/Llama-2-7b-hf"

    # 启动 placement group & actor
    pg_tp2 = placement_group([{"GPU": 2}], strategy="PACK")
    pg_tp4 = placement_group([{"GPU": 4}], strategy="PACK")
    ray.get(pg_tp2.ready())
    ray.get(pg_tp4.ready())

    engine_tp2 = VLLMEngine.options(
        placement_group=pg_tp2,
        placement_group_bundle_index=0,
        placement_group_capture_child_tasks=True,
    ).remote(model_name, 2)

    engine_tp4 = VLLMEngine.options(
        placement_group=pg_tp4,
        placement_group_bundle_index=0,
        placement_group_capture_child_tasks=True,
    ).remote(model_name, 4)

    # 初始化路由器
    router = InferenceRouter(engine_tp2, engine_tp4)

    # 启动 REST 服务
    uvicorn.run("your_script_name:app", host="0.0.0.0", port=8000)