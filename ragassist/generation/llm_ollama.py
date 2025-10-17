import ollama
from ..mytypes import LLMResponse
from .llm_base import LLMBase

class LLMOllama(LLMBase):
    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client()

    def generate(self, system_prompt: str, retrievals: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        prompt = f"{system_prompt}\n\n[User Question]\n{user_prompt}"
        # simple CLI call; replace with HTTP if preferred
        print(f"User prompt to LLMOllama: {user_prompt}")
        sp = "you are an assistant to an AI. Your task is to analyze user's input" \
              " and reformulate into clear and concise request for LLM ingestion. Generate just the request." \
              #f"Document sources marked [Source #] below\n {retrievals}"  # system prompt not used in ollama chat
        result = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sp},
                {"role": "user", "content": f"Analyze and reformulate the following question: {user_prompt}"}
            ]
        )
        print(f"Reformulated question: {result['message']['content'].strip()}")
        result = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"context:\n{retrievals}\n\n [Task]: {user_prompt}"}
                ]
        )
        text = result['message']['content'].strip()
        # minimal schema; downstream will extract citations via patterns
        return LLMResponse(answer=text, citations=[], confidence=0.5, mode="answer")
