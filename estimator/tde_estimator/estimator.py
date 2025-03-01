from typing import Tuple

import numpy as np
from cpsd_tde import cpsd_tde
from xcorr_tde import xcorr_tde
from yacs.config import CfgNode as CN

from ..estimator import Estimator


class Three_Ele_TDE_Estimator(Estimator):
    # TODO
    def __init__(self):
        pass


def far_locate(t12: float, t23: float, c: float, K: float, d: float) -> Tuple[float, float]:
    """远场条件下的位置解算 (近似解). 方位角估计值对时延估计值容错相对精确解更高.

    Parameters
    ----------
    t12 : float
        tau12估计时延
    t23 : float
        tau23估计时延
    c : float
        声速 (m/s)
    K : float
        2, 3阵元间基线长度/1, 2阵元间基线长度
    d : float
        1, 2阵元间基线长度 (m)

    Returns
    -------
    Tuple[float, float]
        r (m), theta (度). 距离估计值, 方位角估计值
    """
    theta = np.mean(np.arccos((c * np.array([t12, t23])) / (np.array([1, K]) * d)))
    r = (
        K * (K + 1) * ((d * np.sin(theta)) ** 2)
        / (2 * c * (K * t12 - t23))
    )
    return r, np.rad2deg(theta)


def near_locate(t12: float, t23: float, c: float, K: float, d: float) -> Tuple[float, float]:
    """近场条件下的位置解算 (精确解).

    Parameters
    ----------
    t12 : float
        tau12估计时延
    t23 : float
        tau23估计时延
    c : float
        声速 (m/s)
    K : float
        2, 3阵元间基线长度/1, 2阵元间基线长度
    d : float
        1, 2阵元间基线长度 (m)

    Returns
    -------
    Tuple[float, float]
        r (m), theta (度). 距离估计值, 方位角估计值
    """
    theta2 = np.arccos(
        (c * d**2 * (K**2 * t12 + t23) - c**3 * t12 * t23 * (t12 + t23))
        / (K * (K + 1) * d**3 - c**2 * d * (K * t12**2 + t23**2))
    )
    r2 = (
        (K * (K + 1) * d**2 - c**2 * (K * t12**2 + t23**2))
        / (2 * c * (K * t12 - t23))
    )
    d2 = (K - 1) * d / 2
    r = np.sqrt(r2**2 + d2**2 + 2 * r2 * d2 * np.cos(theta2))
    if d2 == 0:
        theta = theta2
    else:
        theta = np.arccos((r**2 + d2**2 - r2**2) / (2 * r * d2))
    return r, np.rad2deg(theta)


def cw_delay_estimation(x1: np.ndarray, x2: np.ndarray, sig_cfg: CN, up_sample_scale: int) -> float:
    import copy
    from scipy import interpolate
    """基于CW信号的时延估计

    Parameters
    ----------
    x1 : np.ndarray
        信号1序列 (左侧)
    x2 : np.ndarray
        信号2序列 (右侧)
    fs : float
        信号采样频率 (Hz)
    f : float
        CW信号载波频率 (Hz)
    period : float
        CW信号周期 (s)
    T_on : float
        CW信号有效时长 (s)

    Returns
    -------
    float
        时延估计值tau (s)
        """
    t = np.arange(len(x1)) / sig_cfg.fs
    t_new = np.arange(up_sample_scale * len(x1)) / sig_cfg.fs / up_sample_scale
    x1 = interpolate.interp1d(t, x1, kind='cubic', fill_value='extrapolate')(t_new)  # type: ignore
    x2 = interpolate.interp1d(t, x2, kind='cubic', fill_value='extrapolate')(t_new)  # type: ignore
    s_ref = np.cos(2 * np.pi * sig_cfg.f * t_new) * (t_new % sig_cfg.prf < sig_cfg.pulse_width)
    new_sig_cfg = copy.deepcopy(sig_cfg)
    new_sig_cfg.fs = up_sample_scale * new_sig_cfg.fs
    tau1_frac = xcorr_tde(x1, s_ref, new_sig_cfg)
    tau2_frac = xcorr_tde(x2, s_ref, new_sig_cfg)
    return tau1_frac - tau2_frac


def ambiguity_resolution(phi12_frac: float, phi23_frac: float, K: float, d: float) -> Tuple[float, float]:
    """基于虚拟基线法解相位模糊 (基于平行声源模型)

    Parameters
    ----------
    phi12_frac : float
        一个周期范围内的1, 2阵元信号相位差, (phi1 - phi2) / 2pi, 应在[-1/2, 1/2)
    phi23_frac : float
        一个周期范围内的2, 3阵元信号相位差, (phi2 - phi3) / 2pi, 应在[-1/2, 1/2)
    K : float
        2, 3阵元间基线长度/1, 2阵元间基线长度
    d : float
        1, 2阵元间基线长度 (m)

    Returns
    -------
    Tuple[float, float]
        phi12_hat, phi23_hat (无模糊的1, 2阵元信号相位差, 无模糊的2, 3阵元信号相位差).
    """
    # 确保在[-1/2, 1/2)范围内
    phi12_frac, phi23_frac = np.mod(phi12_frac + 0.5, 1) - 0.5, np.mod(phi23_frac + 0.5, 1) - 0.5
    phi43 = phi23_frac - phi12_frac
    # 1. 修正phi43使-1/2 < phi_43 < 1/2
    phi43 = phi43 - (phi43 > 0.5) + (phi43 < -0.5)
    # 2. 由phi43正负修正phi12_frac, phi23_frac
    sign43 = 2 * (phi43 >= 0) - 1
    phi12_frac = phi12_frac + sign43 * (sign43 * np.sign(phi12_frac) == -1)
    phi23_frac = phi23_frac + sign43 * (sign43 * np.sign(phi23_frac) == -1)
    # 3. 解得n12, n23
    n12 = np.round((abs(phi23_frac) - K * abs(phi12_frac)) / (K - 1))
    if n12 < 0:
        n12 = np.round((abs(phi23_frac) - K * abs(phi12_frac) + 1) / (K - 1))
        n23 = n12 + 1
    else:
        n23 = n12
    n12, n23 = (2 * (phi43 >= 0) - 1) * n12, (2 * (phi43 >= 0) - 1) * n23
    return n12 + phi12_frac, n23 + phi23_frac
