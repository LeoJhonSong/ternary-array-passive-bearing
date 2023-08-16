import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter

def far_locate(t12_f, t23_f, c, K, d):  # FIXME: 还是模糊相位角?
    """远场条件下的定位解算

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