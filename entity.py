from math import ceil
from typing import Tuple

import numpy as np
from numba import jit

from rng import Multithreaded_Standard_Normal, generate_perlin_noise_2d
from utils import deg_pol2cart


class CW_Signal:
    def __init__(self, f: float, T: float, T_on: float) -> None:
        self.f = f
        self.T_on = T_on
        self.T = T

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _generate(t: np.ndarray, f: float, T: float, T_on: float):
        # t为矩阵
        return np.cos(2 * np.pi * f * t) * (t % T < T_on)

    def generate(self, t: np.ndarray):
        # t为矩阵
        return self._generate(t, self.f, self.T, self.T_on)


class CW_Source:
    def __init__(self, signal: CW_Signal, r: float, angle: float) -> None:
        self.signal = signal
        self.r = r
        self.angle = angle
        self.position = deg_pol2cart(r, angle)
        self.set_noise_params()

    def init_rng(self, seed: int | None = None) -> None:
        perlin_noise = generate_perlin_noise_2d((256, 256), (8, 8), seed=seed).flatten()
        self.perlin_series = np.interp(perlin_noise, [np.min(perlin_noise), np.max(perlin_noise)], [-1, 1])
        self.add_w = Multithreaded_Standard_Normal(seed=seed).generate

    def set_noise_params(self, add_w_std: float = 0.3, add_perlin_mag: float = 0.5, T_shift: float = 1):
        """设置各噪声系数

        Parameters
        ----------
        add_w_std : float
            加性高斯白噪声标准差, by default 0.5
        add_perlin_mag : float
            加性Perlin噪声最大幅值, by default 0.3
        T_shift : float
            CW信号周期漂移 (增大) 系数, by default 1
        """
        self.add_w_std = add_w_std
        self.add_perlin_mag = add_perlin_mag
        self.signal.T = T_shift * self.signal.T

    def _noise_gen(self, t: np.ndarray):
        t_len = t.shape[-1]
        add_perlin = np.tile(self.perlin_series, ceil(t_len / self.perlin_series.shape[-1]))[:t_len]  # 重复拼接
        return self.add_w_std * self.add_w(t_len) + self.add_perlin_mag * add_perlin

    def signal_gen(self, t: np.ndarray):
        return self.signal.generate(t) + self._noise_gen(t)


class Three_Elements_Array:
    def __init__(self, d: float, K: float) -> None:
        self.K = K
        self.d = d
        d1, d2, d3 = -(K + 1) * d / 2, -(K - 1) * d / 2, (K + 1) * d / 2
        self.d_i = np.array([d1, d2, d3])
        self.set_noise_params()

    def init_rng(self, seed: int | None = None) -> None:
        self.add_w = Multithreaded_Standard_Normal(seed=seed).generate

    def set_noise_params(self, add_w0_std: float = 7e-3, add_w1_std: float = 7e-3, add_w2_std: float = 7e-3):
        self.add_w_std = np.array([add_w0_std, add_w1_std, add_w2_std]).reshape(-1, 1)

    def noise_gen(self, t: np.ndarray):
        return self.add_w_std * self.add_w(t.shape)


class Array_Signals:
    def __init__(self, source: CW_Source, array: Three_Elements_Array, c: float) -> None:
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
        self.source = source
        self.array = array
        self.c = c
        self.position = np.array(np.zeros(2))
        self.t_last = 0
        self.v_orth_last = np.array([1, 0])[:, np.newaxis]

    def init_rng(self, seed: int | None = None):
        self.source.init_rng(seed)
        self.array.init_rng(seed)

    def set_ideal(self):
        self.source.set_noise_params(0, 0, 1)
        self.array.set_noise_params(0, 0, 0)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _next_rit(self_position: np.ndarray, velocity: np.ndarray, delta_t: np.ndarray, source_position: np.ndarray, d_vec_i: np.ndarray):
        self_position, velocity, delta_t = np.broadcast_arrays(self_position[:, np.newaxis], velocity[:, np.newaxis], delta_t[np.newaxis, :])  # shape: (2, t_len)
        position_t = self_position + velocity * delta_t
        source_position, position_t_expanded, d_vec_i = np.broadcast_arrays(
            source_position[:, np.newaxis, np.newaxis],  # shape: (2, 1, 1)
            position_t[:, np.newaxis, :],  # shape: (2, 1, t_len)
            d_vec_i[:, :, np.newaxis]  # shape: (2, 3, 1)
        )  # shape: (2, 3, t_len)
        r_xy_i_t = source_position - position_t_expanded - d_vec_i
        return np.sqrt(np.sum(r_xy_i_t ** 2, axis=0)), position_t[:, -1]  # shape: (3, t_len), (2,)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _next_xit(r_i_t: np.ndarray, source_signal: np.ndarray, array_noise: np.ndarray):
        # TODO: 水声信道的乘性噪声/低通效应还是可以有?
        return 1 / r_i_t * source_signal + array_noise

    def next(self, t: np.ndarray, velocity: np.ndarray | Tuple[float, float]) -> np.ndarray:
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
        u_orth = np.array([velocity[1], -velocity[0]])[:, np.newaxis] / np.linalg.norm(velocity) if not np.all(velocity == 0) else self.v_orth_last  # 顺时针旋转90度的单位向量
        r_i_t, self.position = self._next_rit(
            self.position,
            velocity,
            t - self.t_last,
            self.source.position,
            u_orth @ self.array.d_i[np.newaxis, :]  # shape: (2, 3)
        )
        x_i_t = self._next_xit(
            r_i_t,
            self.source.signal_gen(t - r_i_t / self.c),
            self.array.noise_gen(t)
        )
        self.t_last = t[-1]
        self.v_orth_last = u_orth
        # TODO: 加入对数值的量化按-5~+5V, 16位进行量化
        return x_i_t
