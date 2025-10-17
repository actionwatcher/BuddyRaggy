from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from ..mytypes import Chunk

class Embedder:
    def __init__(self, text_model: str, code_model: str, device: str = "auto"):
        self.text_model = SentenceTransformer(text_model, device=device)
        self.code_model = SentenceTransformer(code_model, device=device)

    def embed_batch(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        texts, ids = [], []
        for c in chunks:
            texts.append(c.text)
            ids.append(c.id)
        # route to model by type (code vs text)
        # simple split:
        embs = []
        for c in chunks:
            m = self.code_model if c.type == "code" else self.text_model
            embs.append(m.encode(c.text, normalize_embeddings=True))
        return {"ids": ids, "embeddings": np.vstack(embs)}
