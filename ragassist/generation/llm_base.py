from ..types import LLMResponse

class LLMBase:
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        raise NotImplementedError
