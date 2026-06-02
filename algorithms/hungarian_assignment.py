from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_defenders_to_intruders(defenders: np.ndarray, intruders: np.ndarray) -> list[tuple[int, int]]:
    cost = np.linalg.norm(defenders[:, None, :] - intruders[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))
