from collections import deque
from typing import List, Dict

class SessionMemory:
    def __init__(self, max_turns: int = 20):
        self.buf = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.buf.append({"role": role, "content": content})

    def window(self, n: int = 6) -> List[Dict]:
        return list(self.buf)[-n:]
