from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CooperationMode = Literal["intercept", "block", "hold", "regroup"]


@dataclass(frozen=True)
class HighLevelAction:
    target_selection: int
    defense_zone_selection: np.ndarray
    cooperation_mode: CooperationMode


class HighLevelPolicy:
    def __init__(self, high_level_interval: int = 10):
        self.high_level_interval = high_level_interval
        self.current_action: HighLevelAction | None = None

    def select_option(self, step: int) -> str:
        return self.select_action(step, {}).cooperation_mode

    def select_action(self, step: int, info: dict) -> HighLevelAction:
        if self.current_action is not None and step % self.high_level_interval != 0:
            return self.current_action
        threat_scores = info.get("threat_scores", np.zeros(1, dtype=np.float32))
        intruder_positions = info.get("intruder_positions", np.zeros((1, 2), dtype=np.float32))
        protected_asset = info.get("protected_asset_position", np.array([500.0, 500.0], dtype=np.float32))
        target = int(np.argmax(threat_scores)) if len(threat_scores) else 0
        mode: CooperationMode = "intercept"
        if len(threat_scores) and float(np.max(threat_scores)) < 0.25:
            mode = "hold"
        elif len(threat_scores) and float(np.max(threat_scores)) > 0.75:
            mode = "block"
        zone = (intruder_positions[target] + protected_asset) / 2.0 if len(intruder_positions) else protected_asset
        self.current_action = HighLevelAction(target, zone.astype(np.float32), mode)
        return self.current_action
