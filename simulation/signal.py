from typing import Tuple
import numpy as np
from .perlin import generate_perlin_noise_2d


class Array_Signals:
    def __init__(self, c: float, f: float, T_on: float, T: float, r: np.ndarray, d: Tuple[float, float, float]) -> None:
        """行进中三元线阵数字信号仿真数据

        Parameters
        ----------
        c : float
            声速 (m/s)
        f : float
            CW信号载波频率 (Hz)
        T_on : float
            CW信号脉冲宽度 (s)
        T : float
            CW信号周期 (s)
        r : np.ndarray
            以线阵中心为原点的声源坐标 (m, m)
        d : Tuple[float, float, float]
            以线阵中心为原点的三阵元x轴坐标 (m), 1阵元位于负半轴
        """
        self.t_last = 0
        self.v_orth_last = np.matrix([[1], [0]])
        self.c = c
        self.f = f
        self.T_on = T_on
        self.T = T
        self.r = np.matrix([[r[0]], [r[1]]])  # 列向量
        self.d = d  # 三个阵元x方向坐标

    def init_rng(self, seed: int | None) -> None:
        """初始化随机数生成器

        Parameters
        ----------
        seed : int | None
            用于计算7个随机数生成器种子的数值, None则使用默认随机数生成器
        """
        perlin_noise = generate_perlin_noise_2d((256, 256), (8, 8), seed=seed).flatten()
        self.perlin_series = np.interp(perlin_noise, [np.min(perlin_noise), np.max(perlin_noise)], [-1, 1])
        if seed is None:
            self.w = [np.random.default_rng().standard_normal for _ in range(7)]
        else:
            self.w = [np.random.default_rng(10000 * i * seed).standard_normal for i in range(1, 8)]

    def set_params(self, add_w: float, add_perlin: float, cw_t: float, mag_w: float, phase_w: float) -> None:
        """设置各噪声系数

        Parameters
        ----------
        add_w : float
            加性高斯白噪声系数
        add_perlin : float
            加性Perlin噪声系数
        cw_t : float
            CW信号周期增大系数
        mag_w : float
            乘性高斯白噪声系数
        phase_w : float
            相位高斯白噪声系数
        """
        self.k = [add_w, add_perlin, 1 + cw_t, mag_w, phase_w]

    def _source(self, t):
        return np.cos(2 * np.pi * self.f * t) * (t % (self.k[2] * self.T) < self.T_on)

    def _noise(self, t_len: int):
        perlin_noise = np.tile(self.perlin_series, 1 + int(t_len / len(self.perlin_series)))[:t_len]  # 重复拼接
        return self.k[0] * self.w[0](t_len) + self.k[1] * perlin_noise

    def next(self, t: float | np.ndarray, velocity: np.ndarray | Tuple[float, float]) -> np.ndarray:
        """获取下一组指定时刻信号

        Parameters
        ----------
        t : float | np.ndarray
            指定时间序列 (s)
        velocity : np.ndarray | Tuple[float, float]
            上一时刻线阵中心坐标下速度向量 (m/s, m/s)

        Returns
        -------
        np.ndarray
            阵元1, 2, 3处数字信号序列
        """
        velocity = np.array(velocity) if isinstance(velocity, tuple) else velocity
        t_len = 1 if isinstance(t, float) else len(t)
        r_t = self.r - np.matrix(velocity).T @ np.matrix(t - self.t_last)
        v_orth = np.matrix([[velocity[1]], [-velocity[0]]]) if not np.all(velocity == 0) else self.v_orth_last  # 顺时针旋转90度
        r1_t = np.linalg.norm(r_t - self.d[0] * v_orth, axis=0)
        r2_t = np.linalg.norm(r_t - self.d[1] * v_orth, axis=0)
        r3_t = np.linalg.norm(r_t - self.d[2] * v_orth, axis=0)
        x1 = (1 + r1_t**0.5 / 45 * self.k[3] * self.w[1](t_len)) / r1_t * (self._source(t - r1_t / self.c - r1_t**0.5 / 45 * self.k[4] * self.w[4](t_len)) + r1_t**0.5 / 45 * self._noise(t_len))
        x2 = (1 + r2_t**0.5 / 45 * self.k[3] * self.w[2](t_len)) / r2_t * (self._source(t - r2_t / self.c - r2_t**0.5 / 45 * self.k[4] * self.w[5](t_len)) + r2_t**0.5 / 45 * self._noise(t_len))
        x3 = (1 + r3_t**0.5 / 45 * self.k[3] * self.w[3](t_len)) / r3_t * (self._source(t - r3_t / self.c - r3_t**0.5 / 45 * self.k[4] * self.w[6](t_len)) + r3_t**0.5 / 45 * self._noise(t_len))
        self.t_last = t if isinstance(t, float) else t[-1]
        self.v_orth_last = velocity if not np.all(velocity == 0) else self.v_orth_last
        self.r = r_t if isinstance(t, float) else r_t[:, -1]
        # TODO: 加入对数值的量化按-5~+5V, 16位进行量化
        return np.array([x1, x2, x3])
