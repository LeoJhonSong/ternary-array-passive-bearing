import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter


def ambiguity_resolution(phi12_frac: np.ndarray, phi23_frac: np.ndarray, K: float, d: float):
    phi43 = phi23_frac - phi12_frac
    # 1. 修正phi43使-1/2 < phi_43 < 1/2
    phi43 = phi43 - (phi43 > 0.5).astype(int) + (phi43 < -0.5).astype(int)
    # 2. 由phi43正负修正phi12_frac, phi23_frac # FIXME: phi43=0?
    sign43 = np.sign(phi43)
    phi12_frac = phi12_frac + sign43 * (sign43 * np.sign(phi12_frac) == -1).astype(int)
    phi23_frac = phi23_frac + sign43 * (sign43 * np.sign(phi23_frac) == -1).astype(int)
    # 3. 解得n12, n23
    n12 = np.zeros(phi12_frac.shape, dtype=int)
    n23 = np.zeros(phi23_frac.shape, dtype=int)
    for i in range(len(phi12_frac)):
        n12[i] = np.round((phi23_frac[i]  - K * phi12_frac[i]) / (K - 1))
        if phi43[i] > 0 and n12[i] < 0:
            n12[i] = np.round((phi23_frac[i]  - K * phi12_frac[i] + 1) / (K - 1))
            n23[i] = n12[i] + 1
        elif phi43[i] < 0 and n12[i] > 0:
            n12[i] = np.round((phi23_frac[i]  - K * phi12_frac[i] - 1) / (K - 1))
            n23[i] = n12[i] - 1
        else:
            n23[i] = n12[i]
    return n12 + phi12_frac, n23 + phi23_frac


def far_locate(t12_f, t23_f, c, K, d):  # FIXME: 还是模糊相位角?
    """远场条件下的位置解算

    Parameters
    ----------
    t12_f : float
        tau12模糊时延
    t23_f : float
        tau23模糊时延

    Returns
    -------
    r_e : float
        距离估计值
    theta_e : float
        方位角估计值
    """
    t12_e, t23_e = t12_f, t23_f  # TODO: 解模糊
    t13_e = t12_e + t23_e
    theta_e = np.arccos((c * t13_e) / ((K + 1) * d))
    r_e = (
        K * (K + 1) * ((d * np.sin(theta_e)) ** 2)
        / (2 * c * (K * t12_e - t23_e))
    )
    return r_e, theta_e

def time_delay_estimation(s1: np.ndarray, s2: np.ndarray, fs: float) -> float:
    """基于互相关的时延估计

    Parameters
    ----------
    s1 : np.ndarray
        信号序列1 (左侧)
    s2 : np.ndarray
        信号序列2 (右侧)
    fs : float
        采样频率

    Returns
    -------
    float
        时延估计值
    """
    corr = signal.correlate(s1, s2)
    # corr = corr / np.max(corr)
    N = len(s1)  # 两个序列应当一样长所以用同一个长度
    lags = signal.correlation_lags(N, N)
    return lags[np.argmax(corr)] / fs