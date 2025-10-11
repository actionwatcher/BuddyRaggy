from fastapi import FastAPI
from pydantic import BaseModel
from .cli import load_cfg
from .index.vector_store import VectorStore
from .index.bm25_store import BM25Store
from .retrieval.retriever import Retriever
from .generation.context_assembler import ContextAssembler
from .generation.llm_ollama import LLMOllama

app = FastAPI()
cfg = load_cfg("configs/default.yaml")
vec = VectorStore(cfg["vector_store"]["collection"], cfg["project"]["index_dir"])
bm25 = BM25Store(cfg["project"]["index_dir"]) if cfg["bm25"]["enabled"] else None
retr = Retriever(vec, bm25, embed_model=cfg["embedding"]["text_model"], alpha_dense=cfg["retrieval"]["alpha_dense"], rrf=cfg["retrieval"]["rrf"])
llm = LLMOllama(cfg["llm"]["model"])
assembler = ContextAssembler()

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask(q: Query):
    hits = retr.retrieve(q.query, k=cfg["retrieval"]["top_k"])
    ctx = assembler.build(q.query, hits)
    system = cfg["prompting"]["system_message"]
    resp = llm.generate(system, f"[Context]\n{ctx}\n\n[Task] {q.query}", cfg["llm"]["max_output_tokens"], cfg["llm"]["temperature"])
    return {"answer": resp.answer, "hits": [{"file": h.chunk.file_path, "pos": h.chunk.position, "score": h.score} for h in hits]}
