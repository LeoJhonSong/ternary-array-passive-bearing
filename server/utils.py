from typing import Tuple
import numpy as np


def deg2vec(deg: float, length: float) -> np.ndarray:
    return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))]) * length


def calc_real(c: float, d: Tuple[float, float, float], vel_angle: float, position: np.ndarray, S: np.ndarray) -> Tuple[float, float, float, float]:
    v_orth = deg2vec(vel_angle - 90, 1)
    r_vec = S - position
    r123 = np.linalg.norm(
        np.vstack((
            r_vec - d[0] * v_orth,
            r_vec - d[1] * v_orth,
            r_vec - d[2] * v_orth
        )),
        axis=1
    )
    t12, t23 = (r123[:-1] - r123[1:]) / c
    r = float(np.linalg.norm(r_vec))
    angle = np.rad2deg(np.arctan(r_vec[1] / r_vec[0])) - (vel_angle - 90)
    return t12, t23, r, angle
