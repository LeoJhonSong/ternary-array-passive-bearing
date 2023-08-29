import numpy as np
from .perlin import generate_perlin_noise_2d


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

    np.random.seed(0)  # 确保每次随机种子一样
    noise_template = generate_perlin_noise_2d((256, 256), (8, 8))
    noise_template = np.interp(noise_template, [np.min(noise_template), np.max(noise_template)], [-1, 1])
    # plt.imshow(noise_template, cmap='gray', interpolation='lanczos')
    # plt.show()
    noise = noise_template.flatten()
    noise = np.tile(noise, 1 + int(T * fs / (len(noise))))[:int(T * fs)]
    return 1 / r * (np.cos(2 * np.pi * f * (np.arange(0, T, 1 / fs) - r / c)) + 0.7 * noise + 0.6 * rng.standard_normal(int(T * fs)))  # TODO: 可选perlin噪声
    # return 1 / r * (np.cos(2 * np.pi * f * (np.arange(0, T, 1 / fs) - r / c)) + 0.4 * rng.standard_normal(int(T * fs)))