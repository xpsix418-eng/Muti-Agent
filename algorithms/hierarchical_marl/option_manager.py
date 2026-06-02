from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptionManager:
    options: list[str]
    option_horizon: int

    def active_option(self, step: int) -> str:
        index = (step // self.option_horizon) % len(self.options)
        return self.options[index]

    def should_update(self, step: int) -> bool:
        return step % self.option_horizon == 0
