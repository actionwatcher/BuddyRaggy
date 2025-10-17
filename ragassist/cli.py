import typer, yaml
from .ingestion.file_loader import FileLoader
from .ingestion.preprocess import extract_text
from .ingestion.chunker import Chunker
from .ingestion.embedder import Embedder
from .index.vector_store import VectorStore
from .index.bm25_store import BM25Store
from .retrieval.retriever import Retriever
from .generation.context_assembler import ContextAssembler
from .generation.llm_ollama import LLMOllama

app = typer.Typer()

def load_cfg(path: str):
    with open(path) as f: return yaml.safe_load(f)

@app.command()
def ingest(config: str = "configs/default.yaml"):
    cfg = load_cfg(config)
    fl = FileLoader(cfg["project"]["root_dir"],
                    cfg["ingestion"]["include_globs"],
                    cfg["ingestion"]["exclude_globs"],
                    cfg["project"]["id"])
    files = fl.load_files()
    ch = Chunker(cfg)

    all_chunks = []
    for fd in files:
        text = extract_text(fd.path, fd.type)
        all_chunks.extend(ch.chunk(text, fd.path, fd.type))

    emb = Embedder(cfg["embedding"]["text_model"], cfg["embedding"]["code_model"], device=cfg["embedding"]["device"])
    vec = VectorStore(cfg["vector_store"]["collection"], cfg["project"]["index_dir"])
    bm25 = BM25Store(cfg["project"]["index_dir"]) if cfg["bm25"]["enabled"] else None

    # batch for memory, simplify:
    batch = 256
    for i in range(0, len(all_chunks), batch):
        chunk_batch = all_chunks[i:i+batch]
        out = emb.embed_batch(chunk_batch)
        vec.add(chunk_batch, out["embeddings"])
        if bm25: bm25.add(chunk_batch)

    typer.echo(f"Ingested {len(all_chunks)} chunks.")

@app.command()
def ask(q: str, config: str = "configs/default.yaml"):
    cfg = load_cfg(config)
    vec = VectorStore(cfg["vector_store"]["collection"], cfg["project"]["index_dir"])
    bm25 = BM25Store(cfg["project"]["index_dir"]) if cfg["bm25"]["enabled"] else None
    retr = Retriever(vec, bm25, embed_model=cfg["embedding"]["text_model"],
                     alpha_dense=cfg["retrieval"]["alpha_dense"], rrf=cfg["retrieval"]["rrf"])
    hits = retr.retrieve(q, k=cfg["retrieval"]["top_k"])
    ctx = ContextAssembler().build(q, hits)
    system = cfg["prompting"]["system_message"]
    llm = LLMOllama(cfg["llm"]["model"])
    resp = llm.generate(system, ctx, q, cfg["llm"]["max_output_tokens"], cfg["llm"]["temperature"])
    print(resp.answer)

@app.command()
def chat(config: str = "configs/default.yaml"):
    cfg = load_cfg(config)
    vec = VectorStore(cfg["vector_store"]["collection"], cfg["project"]["index_dir"])
    bm25 = BM25Store(cfg["project"]["index_dir"]) if cfg["bm25"]["enabled"] else None
    retr = Retriever(vec, bm25, embed_model=cfg["embedding"]["text_model"],
                     alpha_dense=cfg["retrieval"]["alpha_dense"], rrf=cfg["retrieval"]["rrf"])
    system = cfg["prompting"]["system_message"]
    llm = LLMOllama(cfg["llm"]["model"])

    while(True):
        q = input("Task: ")
        if q == "\\bye":
            print("exiting...")
            break
        hits = retr.retrieve(q, k=cfg["retrieval"]["top_k"])
        ctx = ContextAssembler().build(q, hits)
        resp = llm.generate(system, ctx, q, cfg["llm"]["max_output_tokens"], cfg["llm"]["temperature"])
        print(resp.answer)

if __name__ == "__main__":
    app()
