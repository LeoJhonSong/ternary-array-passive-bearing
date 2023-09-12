import numpy as np
from .perlin import generate_perlin_noise_2d


def sig_gen(c: float, f: float, r: float, fs: float, T: float, weights: list, rnd_seed: int) -> np.ndarray:
    """生成距声源r处信号
    # TODO: 参数要更新

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
    # 设置随机数种子
    rng = np.random.default_rng(rnd_seed)

    perlin_noise = generate_perlin_noise_2d((256, 256), (8, 8), rnd_seed=rnd_seed).flatten()
    perlin_noise = np.interp(perlin_noise, [np.min(perlin_noise), np.max(perlin_noise)], [-1, 1])
    perlin_noise = np.tile(perlin_noise, 1 + int(T * fs / len(perlin_noise)))[:int(T * fs)]  # 重复拼接
    noise_sources = np.vstack((
        rng.standard_normal(int(T * fs)),
        perlin_noise,
    ))
    src_noise = (np.array([weights]) @ noise_sources).flatten()
    # TODO: 缺少自艇噪声
    period = 1
    T_on = 10e-3
    t = np.arange(0, T, 1 / fs)
    sig = np.cos(2 * np.pi * f * (t - r / c)) * ((t - r / c) % period < T_on)
    return 1 / r * (sig + src_noise)