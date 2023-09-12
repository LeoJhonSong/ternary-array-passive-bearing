from typing import Literal, Tuple
import numpy as np
from scipy import signal


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
        _description_
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


def far_locate(t12_e, t23_e, c, K, d):
    """远场条件下的位置解算

    Parameters
    ----------
    t12_e : float
        tau12估计时延
    t23_e : float
        tau23估计时延

    Returns
    -------
    r_e : float
        距离估计值
    theta_e : float
        方位角估计值
    """
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


def time_delay_estimation(x1: np.ndarray, x2: np.ndarray, f: float, fs: float, method: Literal['cpsd', 'xcorr'] = 'xcorr') -> float:
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
