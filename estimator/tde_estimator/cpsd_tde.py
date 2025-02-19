import numpy as np
from yacs.config import CfgNode as CN


def cpsd_tde(x1: np.ndarray, x2: np.ndarray, sig_cfg: CN) -> float:
    """基于单频点互功率谱的时延估计.
    由于使用gcd寻找FFT最大可用频率采样步长, f与fs应为整数

    Parameters
    ----------
    x1 : np.ndarray
        信号1序列 (左侧)
    x2 : np.ndarray
        信号2序列 (右侧)
    sig_cfg: yacs.config.CfgNode
        信号参数配置

    Returns
    -------
    float
        时延估计值tau (s)
    """
    # TODO: 1. 在有信号的区间以多个窗计算时延差 2. 最好选附近多个频点取平均/算斜率
    f, fs = int(sig_cfg.f), int(sig_cfg.fs)
    f_step = np.gcd(f, fs)
    X1_f = np.fft.rfft(x1, fs // f_step)[f // f_step]
    X2_f = np.fft.rfft(x2, fs // f_step)[f // f_step]
    # 由于tau12 = tau1 - tau2, 这里需要负号
    return -np.angle(X1_f * np.conj(X2_f)) / (2 * np.pi * f)  # type: ignore
