from __future__ import annotations

import numpy as np

from algorithms.hierarchical_marl.high_level_policy import HighLevelAction


class LowLevelPolicy:
    def act(self, high_level_action: HighLevelAction, defender_positions: np.ndarray, defender_velocities: np.ndarray, intruder_positions: np.ndarray, protected_asset_position: np.ndarray) -> np.ndarray:
        if len(intruder_positions) == 0:
            target = protected_asset_position
        elif high_level_action.cooperation_mode == "regroup":
            target = defender_positions.mean(axis=0)
        elif high_level_action.cooperation_mode == "hold":
            target = protected_asset_position
        elif high_level_action.cooperation_mode == "block":
            target = high_level_action.defense_zone_selection
        else:
            target = intruder_positions[min(high_level_action.target_selection, len(intruder_positions) - 1)]
        directions = target[None, :] - defender_positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        desired = directions / np.maximum(norms, 1e-6)
        return np.clip(desired - 0.05 * defender_velocities, -1.0, 1.0).astype(np.float32)
