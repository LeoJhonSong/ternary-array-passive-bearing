from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import gridspec

if TYPE_CHECKING:
    from entity import Snapshot_Generator


def deg_pol2cart(rho: float, angle: float) -> np.ndarray:
    theta = np.deg2rad(angle)
    return rho * np.array([np.cos(theta), np.sin(theta)])


def analysis(sig: 'Snapshot_Generator', tau12_hat: float, tau23_hat: float, r_hat: float, angle_hat: float, vel_angle: float):
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


def tf_plot(x: np.ndarray, fs: float, fmax: float = 80e3, NFFT=256, noverlap_scale: float = 0.5):
    """绘制时域波形和短时傅里叶变换时频图

    Parameters
    ----------
    x : np.ndarray
        一维信号
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
    figsize : tuple, optional
        _description_, by default (40, 20)
    fontsize : int, optional
        _description_, by default 32
    """
    # TODO: 先高通滤波再汉明窗: http://mirlab.org/jang/books/audiosignalprocessing/speechFeatureMfcc_chinese.asp?title=122%25
    if x.ndim == 1:
        x = x.reshape(1, -1)
    f, t_seg, Zxx = signal.stft(x, fs)
    Zxx = np.abs(Zxx[:, f < fmax, :])
    f = f[f < fmax] / 1e3
    t = np.arange(x.shape[1]) / fs
    channels = x.shape[0]
    xmin, xmax = float(np.min(x)), float(np.max(x))

    plt.figure(figsize=(10, 3 * channels))
    gs = gridspec.GridSpec(2 * channels, 1, height_ratios=[1, 2] * channels)

    for c in range(channels):
        ax1 = plt.subplot(gs[2 * c])
        ax1.plot(t, x[c])
        ax1.set_xlim((t[0], t[-1]))
        ax1.set_ylim((xmin, xmax))
        ax1.set_xticklabels([])
        ax1.set_ylabel('Signal')
        ax2 = plt.subplot(gs[2 * c + 1])
        ax2.pcolormesh(t_seg, f, Zxx[c], shading='gouraud', cmap='gnuplot')
        ax2.set_ylabel('Frequency (kHz)')
        if c != channels - 1:
            ax2.set_xticklabels([])
            ax2.set_xlabel('Time (s)')

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
