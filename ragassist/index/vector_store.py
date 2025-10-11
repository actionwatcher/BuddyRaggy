from chromadb import Client
from chromadb.config import Settings
import numpy as np
from typing import List
from ..types import Chunk

class VectorStore:
    def __init__(self, collection: str, persist_dir: str):
        self.client = Client(Settings(is_persistent=True, persist_directory=persist_dir))
        self.col = self.client.get_or_create_collection(collection)

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.col.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[{
                "file_path": c.file_path, "type": c.type, "position": c.position
            } for c in chunks],
            documents=[c.text for c in chunks],
        )

    def query(self, q_emb: np.ndarray, k: int = 8):
        res = self.col.query(query_embeddings=[q_emb.tolist()], n_results=k)
        return res
