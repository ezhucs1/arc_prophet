import os
from openai import OpenAI

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
            api_key="EMPTY",
        )
    return _client

def get_model() -> str:
    return os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
