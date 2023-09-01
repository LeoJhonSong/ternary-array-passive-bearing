from typing import Literal
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter


def ambiguity_resolution(phi12_frac: np.ndarray, phi23_frac: np.ndarray, K: float, d: float):
    # TODO: 1. 缺少文档
    # TODO: 2. 一次应该只需要输入一个值
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
        # TODO: 这里的条件可以按绝对值整合
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

def _time_delay_estimation_xcorr(x1: np.ndarray, x2: np.ndarray, f: float, fs: float) -> float:
    """基于互相关的时延估计.
    声源频率f暂时未用到

    Parameters
    ----------
    s1 : np.ndarray
        信号1序列 (左侧)
    s2 : np.ndarray
        信号2序列 (右侧)
    f : int
        声源频率 (Hz)
    fs : float
        信号采样频率 (Hz)

    Returns
    -------
    float
        时延估计值tau (s)
    """
    # TODO: 考虑加个带通滤波
    corr = signal.correlate(x1, x2)
    return signal.correlation_lags(len(x1), len(x2))[np.argmax(corr)] / fs

def _time_delay_estimation_cpsd(x1: np.ndarray, x2: np.ndarray, f: float, fs: float) -> float:
    """基于互功率谱的时延估计.
    由于使用gcd寻找FFT最大可用频率采样步长, f与fs应为整数

    Parameters
    ----------
    x1 : np.ndarray
        信号1序列 (左侧)
    x2 : np.ndarray
        信号2序列 (右侧)
    f : int
        声源频率 (Hz)
    fs : int
        信号采样频率 (Hz)

    Returns
    -------
    float
        时延估计值tau (s)
    """
    f, fs = int(f), int(fs)
    f_step = np.gcd(f, fs)
    X1_f = np.fft.rfft(x1, fs // f_step)[f // f_step]
    X2_f = np.fft.rfft(x2, fs // f_step)[f // f_step]
    # 由于tau12 = tau1 - tau2, 这里需要负号
    return -np.angle(X1_f * np.conj(X2_f)) / (2 * np.pi * f)  # type: ignore

def time_delay_estimation(x1: np.ndarray, x2: np.ndarray, f: float, fs: float, method: Literal['cpsd', 'xcorr'] ='cpsd') -> float:
    """时延估计算法
    可选互相关法或互谱法

    Parameters
    ----------
    x1 : np.ndarray
        信号1序列 (左侧)
    x2 : np.ndarray
        信号2序列 (右侧)
    f : float
        声源频率 (Hz)
    fs : float
        信号采样频率 (Hz)
    method : Literal['cpsd', 'xcorr'], optional
        时延估计方法, by default 'cpsd'

    Returns
    -------
    float
        时延估计值tau (s)
    """
    methods = {
        'cpsd': _time_delay_estimation_cpsd,
        'xcorr': _time_delay_estimation_xcorr
    }
    return methods[method](x1, x2, f, fs)