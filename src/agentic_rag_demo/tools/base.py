from __future__ import annotations

from typing import Protocol


class Tool(Protocol):
    name: str
    description: str

    def run(self, query: str) -> str: ...
