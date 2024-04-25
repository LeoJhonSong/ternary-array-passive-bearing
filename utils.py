from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

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


def tf_plot(x: np.ndarray, fs: float, t: np.ndarray | None = None, f_max: float = 80000, NFFT=8192, noverlap_scale: float = 0.5, f_tick_step: float = 1e4, figsize=(40, 20), fontsize: int = 32):
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

    Returns
    -------
    _type_
        _description_
    """
    # TODO: 先高通滤波再汉明窗: http://mirlab.org/jang/books/audiosignalprocessing/speechFeatureMfcc_chinese.asp?title=122%25
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize

    def hz_to_khz(x, pos):
        'The two args are the value and tick position'
        return '%1.0f' % (x * 1e-3)
    if t is None:
        t = np.arange(x.shape[0]) / fs
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    ax1.plot(t, x)
    ax1.set_ylabel('Signal', fontsize=fontsize)
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, noverlap=int(noverlap_scale * NFFT), Fs=fs, cmap='gnuplot')
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the .image.AxesImage instance representing the data in the plot
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, f_max)
    ax2.set_xticks(np.arange(0, t[-1] + 1, 1))
    ax2.yaxis.set_major_formatter(FuncFormatter(hz_to_khz))  # Change y-axis label format
    ax2.set_yticks(np.arange(0, f_max, f_tick_step))
    ax2.yaxis.set_minor_locator(MultipleLocator(f_tick_step / 2))
    ax2.set_xlabel('Time (s)', fontsize=fontsize)
    ax2.set_ylabel('Frequency (kHz)', fontsize=fontsize)

    print(len(freqs))

    plt.show()
