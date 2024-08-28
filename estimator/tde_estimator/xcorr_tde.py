import numpy as np
from scipy import signal


def xcorr_tde(x1: np.ndarray, x2: np.ndarray, f: float, fs: float) -> float:
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
    corr = signal.correlate(x1, x2, method='fft')
    return signal.correlation_lags(len(x1), len(x2))[np.argmax(corr)] / fs
