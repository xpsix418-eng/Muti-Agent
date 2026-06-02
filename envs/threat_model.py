from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThreatAssessment:
    nearest_distances: np.ndarray
    zone_distances: np.ndarray
    captured: np.ndarray
    breached: np.ndarray


def assess_threats(
    defender_positions: np.ndarray,
    intruder_positions: np.ndarray,
    protected_center: np.ndarray,
    capture_radius: float,
    protected_zone_radius: float,
) -> ThreatAssessment:
    deltas = defender_positions[:, None, :] - intruder_positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    nearest_distances = distances.min(axis=0) if distances.size else np.array([], dtype=np.float32)
    zone_distances = np.linalg.norm(intruder_positions - protected_center, axis=-1)
    return ThreatAssessment(
        nearest_distances=nearest_distances,
        zone_distances=zone_distances,
        captured=nearest_distances <= capture_radius,
        breached=zone_distances <= protected_zone_radius,
    )
