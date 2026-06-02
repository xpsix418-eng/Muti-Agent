from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HighLevelPolicy:
    options: list[str]

    def select_option(self, step: int) -> str:
        return self.options[step % len(self.options)]
