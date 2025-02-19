import numpy as np
from scipy import signal
from yacs.config import CfgNode as CN


def xcorr_tde(x1: np.ndarray, x2: np.ndarray, sig_cfg: CN) -> float:
    """基于互相关的时延估计.
    声源频率f暂时未用到

    Parameters
    ----------
    s1 : np.ndarray
        信号1序列 (左侧)
    s2 : np.ndarray
        信号2序列 (右侧)
    sig_cfg: yacs.config.CfgNode
        信号参数配置

    Returns
    -------
    float
        时延估计值tau (s)
    """
    corr = signal.correlate(x1, x2, method='fft')
    return signal.correlation_lags(len(x1), len(x2))[np.argmax(corr)] / sig_cfg.fs
