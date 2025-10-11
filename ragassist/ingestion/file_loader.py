# ragassist/ingestion/file_loader.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Set
import os

@dataclass
class FileDescriptor:
    path: str
    type: str           # "code" | "pdf" | "md" | "txt" | ...
    modified_ts: float
    project_id: str
    relpath: str

_CODE_EXTS: Set[str] = {".py", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".java", ".rs", ".go"}

def _detect_type(ext: str) -> str:
    if ext in _CODE_EXTS: return "code"
    if ext == ".pdf":     return "pdf"
    if ext == ".md":      return "md"
    return "txt"

def _normalize_exts(items: Iterable[str]) -> Set[str]:
    """
    Accepts things like ['.pdf', 'pdf', '*.pdf', '**/*.pdf'] and normalizes to {'.pdf'}.
    Non-ext patterns are ignored.
    """
    out: Set[str] = set()
    for it in items or []:
        s = it.strip().lower()
        if not s:
            continue
        # keep only the last '.ext' piece if any
        if "." in s:
            ext = "." + s.split(".")[-1].lstrip("*").lstrip("/")
            # guard against weird inputs like "**/"
            if len(ext) > 1 and all(ch.isalnum() or ch in {'.'} for ch in ext):
                out.add(ext)
    return out

class FileLoader:
    def __init__(self, root_dir: str, includes: List[str], excludes: List[str], project_id: str, follow_symlinks=False):
        self.root = Path(root_dir).resolve()
        # Match by extension ONLY
        self.include_exts: Set[str] = _normalize_exts(includes) or {
            ".py", ".cpp", ".c", ".h", ".hpp", ".md", ".pdf", ".txt"
        }
        # Excludes here remain path-based substring checks for simplicity
        self.exclude_tokens: Set[str] = {".git", "node_modules"} | {e.strip() for e in (excludes or []) if e.strip()}
        self.project_id = project_id
        self.follow_symlinks = follow_symlinks

    def _is_excluded(self, rel: Path) -> bool:
        # simple, fast: skip if any token segment matches an excluded token
        parts = {p for p in rel.parts}
        return any(tok in parts for tok in self.exclude_tokens)

    def _allowed_ext(self, p: Path) -> bool:
        return p.suffix.lower() in self.include_exts

    def load_files(self) -> List[FileDescriptor]:
        files: List[FileDescriptor] = []
        it = self.root.rglob("*") if self.follow_symlinks else (p for p in self.root.rglob("*") if not p.is_symlink())
        for p in it:
            if not p.is_file():
                continue
            rel = p.relative_to(self.root)
            if self._is_excluded(rel):
                continue
            if not self._allowed_ext(p):
                continue
            ext = p.suffix.lower()
            files.append(FileDescriptor(
                path=str(p),
                type=_detect_type(ext),
                modified_ts=p.stat().st_mtime,
                project_id=self.project_id,
                relpath=str(rel),
            ))
        return files
