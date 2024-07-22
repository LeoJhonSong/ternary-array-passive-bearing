from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib import gridspec

if TYPE_CHECKING:
    from entity import Array_Data_Sampler


def deg_pol2cart(rho: float, angle: float) -> np.ndarray:
    theta = np.deg2rad(angle)
    return rho * np.array([np.cos(theta), np.sin(theta)])


def analysis(sig: 'Array_Data_Sampler', tau12_hat: float, tau23_hat: float, r_hat: float, angle_hat: float, vel_angle: float):
    u_orth = np.expand_dims(deg_pol2cart(1, vel_angle - 90), axis=1)
    r_i = np.linalg.norm(
        np.expand_dims(sig.source.position, axis=1) - u_orth @ np.matrix(sig.array.d_i),
        axis=0
    )
    tau12_23 = -np.diff(r_i) / sig.c
    err12_23 = np.array([tau12_hat, tau23_hat]) / tau12_23 - 1
    err_r = r_hat / sig.source.r - 1
    angle_unbiased = angle_hat - 90 + vel_angle
    err_angle = (sig.source.angle - angle_unbiased) / (vel_angle - sig.source.angle)
    return pd.DataFrame(
        {
            'angle': [sig.source.angle, angle_unbiased, angle_unbiased - sig.source.angle, err_angle],
            'tau12': [tau12_23[0], tau12_hat, tau12_hat - tau12_23[0], err12_23[0]],
            'tau23': [tau12_23[1], tau23_hat, tau23_hat - tau12_23[1], err12_23[1]],
            'r': [sig.source.r, r_hat, r_hat - sig.source.r, err_r],
        },
        index=['real', 'estimation', 'abs_error', 'rel_error']
    )


def rfft_plot(x: np.ndarray, fs: float, fc: float | None = None, bandwidth: float | None = None):
    """绘制RFFT频谱图"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(x.shape[1], 1 / fs)
    channels = x.shape[0]
    fig, axs = plt.subplots(nrows=channels, sharex=True)
    if channels == 1:
        axs = [axs]
    Xmax = np.max(X)
    if fc is not None and bandwidth is not None:
        f = f / 1e3
        fc, bandwidth = fc / 1e3, bandwidth / 1e3
        fmin, fmax = fc - bandwidth / 2, fc + bandwidth / 2
        Xmax = np.max(X[:, (f < fmax) & (fmin < f)])
        Xmin = np.min(X[:, (f < fmax) & (fmin < f)])
    for c in range(channels):
        axs[c].plot(f, X[c])
        if fc is not None and bandwidth is not None:
            axs[c].set_xlim((fmin, fmax))
            axs[c].set_ylim((Xmin, Xmax))
        else:
            axs[c].set_ylim(top=Xmax)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.xlabel('Frequency (kHz)')
    plt.show()


def tf_plot(x: torch.Tensor, fs: float, fmax: float = 80e3, NFFT=256, noverlap_scale: float = 0.5, log_scale: bool = True, size_scale=1):
    """绘制时域波形和短时傅里叶变换时频图

    Parameters
    ----------
    x : torch.Tensor
        多个信号波形
    fs : float
        采样频率
    t : np.ndarray | None, optional
        _description_, by default None
    f_max : float, optional
        时频图显示的最大频率, by default 80k
    NFFT : int, optional
        当中心频率较高时选较大的NFFT, 否则尽量小以增大时域分辨率, by default 8192
    noverlap_scale : int, optional
        先固定NFFT, 调整noverlap_scale以增大频域分辨率, by default 0.5
    f_tick_step : float, optional
        _description_, by default 1e4
    log_scale : bool, optional
    """
    # TODO: 先高通滤波再汉明窗: http://mirlab.org/jang/books/audiosignalprocessing/speechFeatureMfcc_chinese.asp?title=122%25
    if x.ndim == 1:
        x = x.unsqueeze(0)

    # 计算STFT
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=NFFT,
        win_length=NFFT,
        hop_length=int(NFFT * (1 - noverlap_scale)),
        center=True,
        pad_mode="reflect",
        power=2.0,
    ).to(x.device)
    spectrogram = spectrogram_transform(x)
    if log_scale:
        spectrogram = spectrogram.log10()

    # 转换频率范围
    fmax_bin = int(fmax / (fs / 2) * spectrogram.shape[1])
    spectrogram = spectrogram[:, :fmax_bin, :]
    f = torch.linspace(0, fmax, fmax_bin) / 1e3
    # 时间轴
    t = torch.arange(x.shape[1]) / fs
    # 扩展时间轴和频率轴以匹配spectrogram的维度
    t_edges = torch.linspace(t[0].item(), t[-1].item(), spectrogram.shape[2]).numpy()
    f_edges = torch.linspace(f[0].item(), f[-1].item(), spectrogram.shape[1]).numpy()
    t = t.numpy()
    f = f.numpy()

    channels = x.shape[0]
    xmin, xmax = float(torch.min(x)), float(torch.max(x))

    x = x.cpu().numpy()
    spectrogram = spectrogram.cpu().numpy()

    plt.figure(figsize=(10 * size_scale, 3 * size_scale * channels))
    gs = gridspec.GridSpec(2 * channels, 1, height_ratios=[1, 2] * channels)

    for c in range(channels):
        ax1 = plt.subplot(gs[2 * c])
        ax1.plot(t, x[c])
        ax1.set_xlim((t[0].item(), t[-1].item()))
        ax1.set_ylim((xmin, xmax))
        ax1.set_xticklabels([])
        ax1.set_ylabel('Voltage (V)')
        ax1.yaxis.set_label_position("right")  # 将y轴标签移动到右侧
        ax1.yaxis.tick_right()  # 将y轴刻度移动到右侧
        ax1.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5)

        ax2 = plt.subplot(gs[2 * c + 1])
        ax2.pcolormesh(t_edges, f_edges, spectrogram[c], shading='gouraud', cmap='viridis')
        ax2.set_ylabel('Frequency (kHz)')
        if c != channels - 1:
            ax2.set_xticklabels([])
        else:
            ax2.set_xticks(np.arange(0, 1.001, 10e-3))
            ax2.set_xlabel('Time (s)')

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
