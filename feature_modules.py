import numpy as np
import torch
from torch import nn


class Spectrogram(nn.Module):
    """band-limited spectrogram,
    accept waveform segments as input, shape: (batch_size, seconds, channels, time)
    """
    def __init__(self, fs, nfft):
        super().__init__()
        self.nfft = nfft
        f_low = 20e3
        f_high = 60e3
        f = np.linspace(0, fs / 2, int(nfft / 2 + 1))
        f_idx = np.argwhere((f > f_low) & (f < f_high)).squeeze()
        self.f_low = f_idx[0]
        self.f_high = f_idx[-1]

    def forward(self, x):
        batch_size, seconds, channels, t_len = x.shape
        spectrogram = torch.stft(
            x.view(-1, t_len),   # 压平batch与时间窗, 并行计算
            n_fft=self.nfft,
            hop_length=int(self.nfft / 4),
            window=torch.hann_window(self.nfft, device=x.device),
            onesided=True,
            return_complex=True
        )  # shape: (batch_size * seconds * channels, freq, time)
        spectrogram = spectrogram[:, self.f_low:self.f_high, :]  # NOTE: 只截取感兴趣频段, 目前未考虑8.8kHz
        spectrogram = spectrogram.reshape(-1, channels, spectrogram.shape[-2], spectrogram.shape[-1])  # shape: (batch_size * seconds, channels, freq_limited, time)
        spectrogram = spectrogram.view(batch_size, seconds, channels, spectrogram.shape[-2], spectrogram.shape[-1])  # shape: (batch_size, seconds, channels, freq_limited, time)
        return spectrogram


class CPSD_Phase_Spectrogram(nn.Module):
    """three element band-limited phase spectrogram of short time cross power spectral density (CPSD),
    accept spectrogram segments as input, shape: (batch_size, seconds, channels, freq_limited, time)
    """
    def __init__(self):
        super().__init__()

    def forward(self, spectrogram):
        batch_size, seconds, channels, freq_len, t_len = spectrogram.shape
        spectrogram = spectrogram.view(batch_size * seconds, channels, freq_len, t_len)  # shape: (batch_size * seconds, channels, freq_limited, time)
        cpsd12_23_13 = torch.stack((
            spectrogram[:, 0, :, :] * torch.conj(spectrogram[:, 1, :, :]),
            spectrogram[:, 1, :, :] * torch.conj(spectrogram[:, 2, :, :]),
            spectrogram[:, 0, :, :] * torch.conj(spectrogram[:, 2, :, :]),
        ), dim=1)  # shape: (batch_size * seconds, 3, freq_limited, time)
        phase12_23_13 = torch.angle(cpsd12_23_13).view(batch_size, seconds, 3, freq_len, t_len)  # shape: (batch_size, seconds, 3, freq_limited, time)
        return phase12_23_13