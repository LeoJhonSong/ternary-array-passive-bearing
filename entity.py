from math import ceil
from typing import Literal, Tuple

import numpy as np
import torch

from rng import generate_perlin_noise_2d
from utils import deg_pol2cart


class CW_Func_Handler:
    def __init__(self, f: float, prf: float, pulse_width: float, device: Literal['cuda', 'cpu'] = 'cpu') -> None:
        self.f = f
        self.pulse_width = pulse_width
        self.prf = prf  # Pulse repetition frequency
        self.pulse_start = np.random.uniform(0, self.prf)  # 随机脉冲起始时间
        self.init_phase = np.random.uniform(-np.pi, np.pi)  # 随机初相位
        self.device = device

    @staticmethod
    def _generate(t: np.ndarray, f: float, init_phase: float, device):
        # t为矩阵
        # return np.cos(2 * np.pi * f * t)
        phase = torch.tensor(2 * np.pi * f * t + init_phase, device=device)
        return torch.cos(phase).cpu().numpy()

    def __call__(self, t: np.ndarray):
        # t为矩阵
        s = np.zeros_like(t)
        t = t + self.pulse_start
        s[t % self.prf < self.pulse_width] = self._generate(t[t % self.prf < self.pulse_width], self.f, self.init_phase, self.device)
        return s

    def t_bound(self, t: np.ndarray):
        pulse_edge = np.diff(((t + self.pulse_start) % self.prf < self.pulse_width).astype(int))
        first = np.where(pulse_edge == 1)[0][0]
        last = np.where(pulse_edge == -1)[0][0] + 1
        return first, last


class CW_Source:
    def __init__(self, signal_func_callback: CW_Func_Handler, r: float, angle: float, seed: int | None = None, device: Literal['cuda', 'cpu'] = 'cpu') -> None:
        self.signal_func_callback = signal_func_callback
        self.r = r
        self.angle = angle
        self.position = deg_pol2cart(r, angle)
        self.set_noise_params()
        # initial rng
        perlin_noise = generate_perlin_noise_2d((256, 256), (8, 8), seed=seed).flatten()
        self.perlin_series = np.interp(perlin_noise, [np.min(perlin_noise), np.max(perlin_noise)], [-1, 1])
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        self.add_w = lambda t_len: torch.randn(t_len, generator=g, device=device).cpu().numpy()

    def set_noise_params(self, add_w_std: float = 1e-7, add_perlin_mag: float = 0.05, T_shift: float = 1):
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
        self.signal_func_callback.prf = T_shift * self.signal_func_callback.prf

    def _noise_gen(self, t: np.ndarray):
        t_len = t.shape[-1]
        if t_len > self.perlin_series.shape[-1]:
            add_perlin = np.tile(self.perlin_series, ceil(t_len / self.perlin_series.shape[-1]))[:t_len]  # 重复拼接
        else:
            add_perlin = self.perlin_series[:t_len]
        return self.add_w_std * self.add_w(t_len) + self.add_perlin_mag * add_perlin

    def signal_gen(self, t: np.ndarray):
        sig = self.signal_func_callback(t)
        if not (self.add_w_std == 0 and self.add_perlin_mag == 0):
            sig += self._noise_gen(t)
        return sig

    def t_bound(self, t: np.ndarray):
        first, last = t.shape[1], 0
        for i in range(t.shape[0]):
            _first, _last = self.signal_func_callback.t_bound(t[i])
            if _first < first:
                first = _first
            if _last > last:
                last = _last
        return first, last


class Three_Elements_Array:
    def __init__(self, d: float, K: float, seed: int | None = None, device: Literal['cuda', 'cpu'] = 'cpu') -> None:
        self.K = K
        self.d = d
        d1, d2, d3 = -(K + 1) * d / 2, -(K - 1) * d / 2, (K + 1) * d / 2
        self.dist_max = self.d * (self.K + 1)  # 最大阵元间距
        self.d_i = np.array([d1, d2, d3])
        self.position = np.array(np.zeros(2))
        self.set_noise_params()
        # initial rng
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        self.add_w = lambda size: torch.randn(size, generator=g, device=device).cpu().numpy()

    def set_noise_params(self, add_w0_std: float = 7e-3, add_w1_std: float = 7e-3, add_w2_std: float = 7e-3):
        self.add_w_std = np.array([add_w0_std, add_w1_std, add_w2_std]).reshape(-1, 1)

    def noise_gen(self, t: np.ndarray):
        return (self.add_w_std * self.add_w((len(self.add_w_std), len(t))))


class Array_Data_Sampler:
    def __init__(self, source: CW_Source, array: Three_Elements_Array, c: float, device: Literal['cuda', 'cpu'] = 'cpu') -> None:
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
        self.t_last = 0
        self.v_orth_last = np.array([1, 0])[:, np.newaxis]
        self.device = device

    def maxlag(self, fs: float) -> int:
        return int(np.ceil(self.array.dist_max / self.c * fs))  # 采样频率下最大滞后量, 即最大时延对应的采样点数

    def set_ideal(self):
        self.source.set_noise_params(0, 0, 1)
        self.array.set_noise_params(0, 0, 0)

    def set_SNR(self, snr_dB, fs, bandwidth):
        offset = (fs / 2 / bandwidth) ** 0.5  # 使信号频带内噪声平均功率为1
        gain = 1 / self.source.r / 10**(snr_dB / 20)  # 当信号频带内噪声平均功率为1时, 使SNR=snr_dB的噪声增益
        deviation = gain * offset
        self.array.set_noise_params(deviation, deviation, deviation)

    @staticmethod
    def _next_rit(array_position, velocity, delta_t, source_position, d_vec_i, device) -> Tuple[np.ndarray, np.ndarray]:
        array_position = torch.tensor(array_position, device=device).unsqueeze(1)
        velocity = torch.tensor(velocity, device=device).unsqueeze(1)
        delta_t = torch.tensor(delta_t, device=device).unsqueeze(0)
        position_t = array_position + velocity * delta_t  # shape: (2, t_len)
        source_position = torch.tensor(source_position[:, np.newaxis, np.newaxis], device=device)  # shape: (2, 1, 1)
        d_vec_i = torch.tensor(d_vec_i[:, :, np.newaxis], device=device)  # shape: (2, 3, 1)
        r_xy_i_t = source_position - position_t.unsqueeze(1) - d_vec_i  # shape: (2, 3, t_len)
        r_xy_i_t = r_xy_i_t.cpu().numpy()
        position_t = position_t.cpu().numpy()
        return np.sqrt(np.sum(r_xy_i_t ** 2, axis=0)), position_t[:, -1]  # shape: (3, t_len), (2,)

    def __call__(self, t: np.ndarray, velocity: np.ndarray | Tuple[float, float]) -> Tuple[np.ndarray, float, float, np.ndarray]:
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
            shape: (3, t_len), 阵元1, 2, 3处数字信号序列
        """
        velocity = np.array(velocity) if isinstance(velocity, tuple) else velocity
        u_orth = np.array([velocity[1], -velocity[0]])[:, np.newaxis] / np.linalg.norm(velocity) if not np.all(velocity == 0) else self.v_orth_last  # 顺时针旋转90度的单位向量
        array2source = self.source.position - self.array.position  # shape: (2,), vector point to source from array center
        r_real = float(np.linalg.norm(array2source))
        angle_real = np.rad2deg(np.arccos(np.dot(array2source, u_orth) / (np.linalg.norm(array2source) * np.linalg.norm(u_orth))))
        r_i_t, self.array.position = self._next_rit(
            self.array.position,
            velocity,
            t - self.t_last,
            self.source.position,
            u_orth @ self.array.d_i[np.newaxis, :],  # shape: (2, 3)
            self.device
        )
        source_signal = torch.tensor(self.source.signal_gen(t - r_i_t / self.c), device=self.device)
        if self.array.add_w_std.sum() == 0:
            x_i_t = (1 / torch.tensor(r_i_t, device=self.device) * source_signal).cpu().numpy()
        else:
            array_noise = torch.tensor(self.array.noise_gen(t), device=self.device)
            x_i_t = (1 / torch.tensor(r_i_t, device=self.device) * source_signal + array_noise).cpu().numpy()
        first, last = self.source.t_bound(t - r_i_t / self.c)
        self.t_last = t[-1]
        self.v_orth_last = u_orth
        # TODO: 加入对数值的量化按-5~+5V, 16位进行量化
        return x_i_t, r_real, angle_real, np.array((t[first], t[last]))
