from typing import List
from ..types import Chunk
import uuid
# optional: tree_sitter integration for better code segmentation

class Chunker:
    def __init__(self, cfg):
        self.cfg = cfg

    def chunk(self, text: str, fpath: str, ftype: str) -> List[Chunk]:
        if ftype == "code":
            return self._code_chunks(text, fpath)
        return self._text_chunks(text, fpath)

    def _make(self, text, fpath, pos, ftype) -> Chunk:
        return Chunk(id=str(uuid.uuid4()), text=text, type=ftype, file_path=fpath, position=pos, meta={})

    def _text_chunks(self, text: str, fpath: str) -> List[Chunk]:
        # simple token-ish windowing; replace with sentence splitter if desired
        max_toks = self.cfg["chunking"]["text"]["max_tokens"]
        overlap = self.cfg["chunking"]["text"]["overlap_tokens"]
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + max_toks, len(words))
            chunk_words = words[start:end]
            chunks.append(self._make(" ".join(chunk_words), fpath, start, "text"))
            start = end - overlap if end - overlap > start else end
        return chunks

    def _code_chunks(self, text: str, fpath: str) -> List[Chunk]:
        # placeholder: function-level split could be done via tree-sitter
        lines = text.splitlines()
        max_lines = 60
        overlap = 8
        chunks, start = [], 0
        while start < len(lines):
            end = min(start + max_lines, len(lines))
            chunks.append(self._make("\n".join(lines[start:end]), fpath, start, "code"))
            start = end - overlap if end - overlap > start else end
        return chunks
