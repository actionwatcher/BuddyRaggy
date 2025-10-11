from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from pathlib import Path
from typing import List
from ..types import Chunk

class BM25Store:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir) / "bm25"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        schema = Schema(id=ID(stored=True, unique=True),
                        content=TEXT(stored=False),
                        file_path=STORED, position=STORED, type=STORED)
        if not (self.index_dir / "MAIN_WRITELOCK").exists() and not any(self.index_dir.iterdir()):
            self.ix = create_in(self.index_dir, schema)
        else:
            self.ix = open_dir(self.index_dir)

    def add(self, chunks: List[Chunk]):
        writer = self.ix.writer()
        for c in chunks:
            writer.add_document(id=c.id, content=c.text, file_path=c.file_path, position=c.position, type=c.type)
        writer.commit()

    def search(self, query: str, k: int = 8):
        with self.ix.searcher() as s:
            qp = QueryParser("content", schema=self.ix.schema)
            q = qp.parse(query)
            hits = s.search(q, limit=k)
            return [{"id": h["id"], "score": h.score} for h in hits]
