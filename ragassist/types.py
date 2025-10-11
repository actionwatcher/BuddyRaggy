from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FileDescriptor:
    path: str
    type: str         # "code" | "pdf" | "md" | "txt" | "email" | ...
    modified_ts: float
    project_id: str

@dataclass
class Chunk:
    id: str
    text: str
    type: str
    file_path: str
    position: int
    meta: Dict

@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    source: str       # "dense" | "bm25" | "fused"

@dataclass
class LLMResponse:
    answer: str
    citations: List[Dict]   # [{"file": "...", "positions": [..]}]
    confidence: float
    mode: str
