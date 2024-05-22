import numpy as np
import torch
from torch import nn


class Spectrogram(nn.Module):
    """band-limited spectrogram,
    accept waveform segments as input, shape: (batch_size, seconds, channels, time)
    """

    def __init__(self, fs, nfft, hop_factor):
        super().__init__()
        self.nfft = nfft
        self.hop_factor = hop_factor
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
            hop_length=int(self.nfft / self.hop_factor),
            window=torch.hann_window(self.nfft, device=x.device),
            onesided=True,
            return_complex=True
        )  # shape: (batch_size * seconds * channels, freq, time)
        spectrogram = spectrogram[:, self.f_low:self.f_high + 1, :]  # NOTE: 只截取感兴趣频段, 目前未考虑8.8kHz
        spectrogram = spectrogram.reshape(-1, channels, spectrogram.shape[-2], spectrogram.shape[-1])  # shape: (batch_size * seconds, channels, freq_limited, time)
        spectrogram = spectrogram.view(batch_size, seconds, channels, spectrogram.shape[-2], spectrogram.shape[-1])  # shape: (batch_size, seconds, channels, freq_limited, time)
        return spectrogram


class Filter(nn.Module):
    def __init__(self, fs, nfft, fc):
        super().__init__()
        f_low = 20e3
        f_high = 60e3
        f = np.linspace(0, fs / 2, int(nfft / 2 + 1))
        self.f = f[(f > f_low) & (f < f_high)]
        self.fc_index = int(np.argwhere(self.f == fc)[0][0])

    def forward(self, x):
        spectrogram_batch = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # shape: (batch_size * seconds, channels, freq_limited, time)
        magnitude_batch = torch.sum(torch.abs(spectrogram_batch), dim=1)  # shape: (batch_size * seconds, freq_limited, time)
        mid_batch = torch.argmax(magnitude_batch[:, 1084, :], dim=1)  # find the time index of the maximum magnitude at the center frequency
        pulse_mag_batch = torch.zeros(magnitude_batch.shape[:2])  # shape: (batch_size * seconds, freq_limited)
        filtered_spectrogram_batch = torch.zeros_like(spectrogram_batch)
        for i, mid in enumerate(mid_batch):
            mid = int(mid)
            pulse_mag_batch[i] = torch.sum(magnitude_batch[i, :, max(0, mid - 8):mid + 8], dim=1)
            noise_mag = torch.mean(magnitude_batch[i, (self.f < 40e3) | (self.f > 44e3)])
            sig_mag = pulse_mag_batch[i, self.fc_index]
            snr = 10 * torch.log10((sig_mag - noise_mag)**2 / noise_mag**2)
            half_band = max(int((12.5 * (snr.cpu() - 20) + 27)), 8)
            filtered_spectrogram_batch[
                :,
                :,
                self.fc_index - half_band:self.fc_index + half_band,
                max(0, mid - 8):min(mid + 8, filtered_spectrogram_batch.shape[-1])
            ] = spectrogram_batch[
                :,
                :,
                1084 - half_band:1084 + half_band,
                max(0, mid - 8):min(mid + 8, filtered_spectrogram_batch.shape[-1])
            ]
        filtered_spectrogram_batch = filtered_spectrogram_batch.reshape(x.shape)
        return filtered_spectrogram_batch


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
