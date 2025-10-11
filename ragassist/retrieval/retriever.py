import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from ..types import RetrievalHit, Chunk

class Retriever:
    def __init__(self, vector_store, bm25_store, embed_model: str, alpha_dense: float = 0.7, rrf: bool = True):
        self.vs = vector_store
        self.bm25 = bm25_store
        self.alpha = alpha_dense
        self.rrf = rrf
        self.embedder = SentenceTransformer(embed_model)

    def _fuse_rrf(self, dense_hits, bm25_hits, k=8):
        # reciprocal rank fusion over chunk IDs
        scores = {}
        for rank, h in enumerate(dense_hits):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (50 + rank)
        for rank, h in enumerate(bm25_hits):
            scores[h["id"]] = scores.get(h["id"], 0.0) + 1.0 / (50 + rank)
        ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:k]
        return [{"id": cid, "score": scores[cid]} for cid in ids]

    def retrieve(self, query: str, k: int = 8) -> List[RetrievalHit]:
        q_emb = self.embedder.encode(query, normalize_embeddings=True)
        dense_res = self.vs.query(q_emb, k=16)
        dense_hits = [{"id": id_, "score": s} for id_, s in zip(dense_res["ids"][0], dense_res["distances"][0])]
        bm25_hits = self.bm25.search(query, k=16) if self.bm25 else []

        if self.rrf and bm25_hits:
            fused = self._fuse_rrf(dense_hits, bm25_hits, k=k)
        else:
            # linear interpolate by normalized ranks
            fused = (dense_hits[:k] if self.alpha >= 0.5 else bm25_hits[:k])

        # map IDs back to full chunks:
        id_to_chunk = {doc_id: Chunk(
            id=doc_id,
            text=doc_text,
            type=meta["type"],
            file_path=meta["file_path"],
            position=meta["position"],
            meta=meta
        ) for doc_id, doc_text, meta in zip(dense_res["ids"][0], dense_res["documents"][0], dense_res["metadatas"][0])}

        hits = []
        for h in fused:
            c = id_to_chunk.get(h["id"])
            if c:
                hits.append(RetrievalHit(chunk=c, score=h["score"], source="fused"))
        return hits[:k]
