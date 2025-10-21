from typing import Dict
from .llm_ollama import LLMOllama
from .llm_gemini import LLMGemini


def get_model(llm_cfg: Dict):
    """Return an LLM instance based on the llm_cfg dict.

    Expected llm_cfg keys: 'backend' and 'model' at minimum.
    """
    backend = llm_cfg.get("backend", "ollama").lower()
    model = llm_cfg.get("model")

    if backend == "ollama":
        return LLMOllama(model)
    if backend in ("gemini", "google", "google-gemini"):
        return LLMGemini(model)

    # default fallback
    return LLMOllama(model)
