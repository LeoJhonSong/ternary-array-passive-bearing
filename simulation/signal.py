import numpy as np


def sig_gen(c: float, f: float, r: float, fs: float, T: float) -> np.ndarray:
    """生成距声源r处信号

    Parameters
    ----------
    c : float
        声速
    f : float
        声源发声频率
    r : float
        信号采样点距声源距离
    fs : float
        采样频率
    T : float
        采样时长

    Returns
    -------
    list
        信号序列s[r, n]
    """
    rng = np.random.default_rng()
    # return 1 / r * (np.cos(2 * np.pi * f * (np.arange(0, T, 1 / fs) - r / c)) + 0.01 * noise)  # TODO: 可选perlin噪声
    return 1 / r * (np.cos(2 * np.pi * f * (np.arange(0, T, 1 / fs) - r / c)) + 0.2 * rng.standard_normal(int(T * fs)))