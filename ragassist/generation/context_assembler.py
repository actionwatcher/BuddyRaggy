from typing import List
from ..types import RetrievalHit

class ContextAssembler:
    def build(self, query: str, hits: List[RetrievalHit], token_budget: int = 6000) -> str:
        # naive budget: just join with file markers
        blocks = []
        for i, h in enumerate(hits, 1):
            header = f"[Source {i}] {h.chunk.file_path} @ {h.chunk.position}\n"
            blocks.append(header + h.chunk.text.strip())
        return "\n\n".join(blocks)
