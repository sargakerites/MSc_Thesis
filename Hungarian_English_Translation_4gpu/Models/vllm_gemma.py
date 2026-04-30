from vllm import LLM, SamplingParams
from pathlib import Path
import os

_llm = None
_sampling_params = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "VLLM_MODEL_PATH",
        PROJECT_ROOT / "models" / "gemma-3-27b-it"
    )
)

LOCAL_MODEL = str(MODEL_PATH)


def model_generation():
    """Initialize and return cached vLLM model along with single and batch text generation functions."""

    global _llm, _sampling_params

    if _llm is None:
        _llm = LLM(
            model=LOCAL_MODEL,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.95,
            max_model_len=16000,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            trust_remote_code=True,
            disable_log_stats=True,
        )

    if _sampling_params is None:
        _sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=3500,
        )

    def complete_one(prompt: str, **kwargs) -> str:
        """Generate a single completion for a given prompt using optional sampling parameter overrides."""

        params = _sampling_params if not kwargs else SamplingParams(
            **{**_sampling_params.__dict__, **kwargs}
        )
        out = _llm.generate([prompt], params)[0]
        return out.outputs[0].text

    def complete_many(prompts: list[str], **kwargs) -> list[str]:
        """Generate completions for multiple prompts in batch using optional sampling parameter overrides."""

        params = _sampling_params if not kwargs else SamplingParams(
            **{**_sampling_params.__dict__, **kwargs}
        )
        outs = _llm.generate(prompts, params)
        return [o.outputs[0].text for o in outs]

    return complete_one, complete_many, _sampling_params