from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from entity import Snapshot_Generator


def deg_pol2cart(rho: float, angle: float) -> np.ndarray:
    theta = np.deg2rad(angle)
    return rho * np.array([np.cos(theta), np.sin(theta)])


def analysis(sig: 'Snapshot_Generator', tau12_hat: float, tau23_hat: float, r_hat: float, angle_hat: float, vel_angle: float):
    u_orth = np.expand_dims(deg_pol2cart(1, vel_angle - 90), axis=1)
    r_i = np.linalg.norm(
        np.expand_dims(sig.source.position, axis=1) - u_orth @ np.matrix(sig.array.d_i),
        axis=0
    )
    tau12_23 = -np.diff(r_i) / sig.c
    err12_23 = np.array([tau12_hat, tau23_hat]) / tau12_23 - 1
    err_r = r_hat / sig.source.r - 1
    angle_unbiased = angle_hat - 90 + vel_angle
    err_angle = (sig.source.angle - angle_unbiased) / (vel_angle - sig.source.angle)
    return pd.DataFrame(
        {
            'angle': [sig.source.angle, angle_unbiased, angle_unbiased - sig.source.angle, err_angle],
            'tau12': [tau12_23[0], tau12_hat, tau12_hat - tau12_23[0], err12_23[0]],
            'tau23': [tau12_23[1], tau23_hat, tau23_hat - tau12_23[1], err12_23[1]],
            'r': [sig.source.r, r_hat, r_hat - sig.source.r, err_r],
        },
        index=['real', 'estimation', 'abs_error', 'rel_error']
    )
