try:
    from google import genai
    from google.genai import types as google_types
except Exception:  # pragma: no cover - optional dependency
    genai = None

from ..mytypes import LLMResponse
from .llm_base import LLMBase


class LLMGemini(LLMBase):
    def __init__(self, model: str):
        self.model = model
        if genai is not None:
            # Typical Gemini clients provide a Client or direct API; adapt as needed
            try:
                self.client = genai.Client()
            except Exception:
                # fallback to module-level client factory if different
                self.client = getattr(genai, "client", None)
        else:
            self.client = None

    def generate(self, system_prompt: str, retrievals: str, user_prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        prompt = f"{system_prompt}\n\n[User Question]\n{user_prompt}"
        print(f"User prompt to LLMGemini: {user_prompt}")

        if self.client is None:
            # Geminiclient not available; return a clear error-like response
            print("LLMGemini client not initialized.")
            return LLMResponse(answer="", citations=[], confidence=0.0, mode="error")

        # Build messages similarly to LLMOllama for compatibility
        sp = """you are an assistant to an AI. Your task is to analyze user's input \
             and reformulate into clear and concise request for LLM ingestion.
              The reformulated request will be used to to compile the answer based on provided context from vector store retrievals."""

        try:
            config = google_types.GenerateContentConfig( temperature=0.0, system_instruction=sp)
            result = self.client.models.generate_content(
                model=self.model,
                config=config,
                contents=user_prompt
            )
            reform = result.candidates[0].content.parts[-1].text

            print(f"Reformulated question: {reform}")

            # Now call the model with system prompt + context
            result2 = self.client.models.generate_content(
                model=self.model,
                config=google_types.GenerateContentConfig( temperature=temperature, system_instruction=system_prompt),
                contents=[f"context:\n{retrievals}\n\n [Task]: {user_prompt}"]
            )
            text = result2.candidates[0].content.parts[-1].text.strip()
            print(text)
            return LLMResponse(answer=text, citations=[], confidence=0.5, mode="answer")
        
        except Exception as exc:
            print(f"LLMGemini error: {exc}")
            return LLMResponse(answer="", citations=[], confidence=0.0, mode="error")
