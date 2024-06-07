from typing import Type, Union

import numpy as np
import torch
from torch import nn


class Spectrogram(nn.Module):
    """band-limited spectrogram,
    accept waveform segments as input, shape: (batch_size, seconds, channels, time)
    """

    def __init__(self, fs, nfft, hop_factor, f_low, f_high):
        super().__init__()
        self.nfft = nfft
        self.hop_factor = hop_factor
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
    def __init__(self, fs, nfft, fc, f_low, f_high):
        """以幅度谱中目标频段的能量和信噪比为依据将背景置零, 输出的是STFT复数张量"""
        super().__init__()
        f = np.linspace(0, fs / 2, int(nfft / 2 + 1))
        self.f = f[(f > f_low) & (f < f_high)]
        self.fc_index = int(np.argwhere(self.f == fc)[0][0])

    def forward(self, x):
        spectrogram_batch = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # shape: (batch_size * seconds, channels, freq_limited, time)
        magnitude_batch = torch.sum(torch.abs(spectrogram_batch), dim=1)  # shape: (batch_size * seconds, freq_limited, time)
        mid_batch = torch.argmax(magnitude_batch[:, self.fc_index, :], dim=1)  # find the time index of the maximum magnitude at the center frequency
        pulse_mag_batch = torch.zeros(magnitude_batch.shape[:2], device=x.device)  # shape: (batch_size * seconds, freq_limited)
        filtered_spectrogram_batch = torch.zeros_like(spectrogram_batch)
        for i, mid in enumerate(mid_batch):
            mid = int(mid)
            pulse_mag_batch[i] = torch.sum(magnitude_batch[i, :, max(0, mid - 20):mid + 20], dim=1)
            noise_mag = torch.mean(magnitude_batch[i, (self.f < 40e3) | (self.f > 44e3)])
            sig_mag = pulse_mag_batch[i, self.fc_index]
            snr = 10 * torch.log10((sig_mag - noise_mag)**2 / noise_mag**2)
            half_band = max(int((6 * (snr.cpu() - 48) + 27)), 8)
            filtered_spectrogram_batch[
                :,
                :,
                max(0, self.fc_index - half_band):self.fc_index + half_band,
                max(0, mid - 8):min(mid + 8, filtered_spectrogram_batch.shape[-1])
            ] = spectrogram_batch[
                :,
                :,
                max(0, self.fc_index - half_band):self.fc_index + half_band,
                max(0, mid - 8):min(mid + 8, filtered_spectrogram_batch.shape[-1])
            ]
        filtered_spectrogram_batch = filtered_spectrogram_batch.reshape(x.shape)
        return filtered_spectrogram_batch


class Crop(nn.Module):
    def __init__(self, fs, nfft, fc, f_low, f_high):
        super().__init__()
        f = np.linspace(0, fs / 2, int(nfft / 2 + 1))
        self.f = f[(f > f_low) & (f < f_high)]
        self.fc_index = int(np.argwhere(self.f == fc)[0][0])

    def forward(self, x):
        spectrogram_batch = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # shape: (batch_size * seq_len, channels, freq_limited, time)
        magnitude_batch = torch.sum(torch.abs(spectrogram_batch), dim=1)  # shape: (batch_size * seq_len, freq_limited, time)
        mid_batch = torch.argmax(magnitude_batch[:, self.fc_index, :], dim=1)  # find the time index of the maximum magnitude at the center frequency
        t_len = 16
        front, end = t_len // 2 - 1, t_len // 2 + 1
        cropped_spectrogram_batch = torch.zeros(spectrogram_batch.shape[0], spectrogram_batch.shape[1], spectrogram_batch.shape[2], t_len, device=x.device, dtype=spectrogram_batch.dtype)
        for i, mid in enumerate(mid_batch):
            mid = int(mid)
            if mid < front:
                cropped_spectrogram_batch[i, :, :, front - mid:] = spectrogram_batch[i, :, :, :mid + end]
            elif mid > spectrogram_batch.shape[3] - end:
                cropped_spectrogram_batch[i, :, :, :front - mid + spectrogram_batch.shape[3]] = spectrogram_batch[i, :, :, mid - front:]
            else:
                cropped_spectrogram_batch[i, :, :, :] = spectrogram_batch[i, :, :, mid - front:mid + end]
        pulse_mag_batch = torch.sum(torch.abs(cropped_spectrogram_batch), dim=(1, 3))  # shape: (batch_size * seq_len, freq_limited)
        noise_mag_batch = torch.mean(pulse_mag_batch[:, (self.f < 41e3) | (self.f > 44e3)], dim=1)  # shape: (batch_size * seq_len,)
        sig_mag_batch = pulse_mag_batch[:, self.fc_index]  # shape: (batch_size * seq_len,)
        snr_batch = 10 * torch.log10((sig_mag_batch - noise_mag_batch)**2 / noise_mag_batch**2)  # shape: (batch_size * seq_len,)
        half_band_batch = self.snr_band_map(snr_batch)
        musk_batch = torch.zeros(spectrogram_batch.shape[0], spectrogram_batch.shape[2], t_len, device=x.device)  # shape: (batch_size * seq_len, freq_limited, t_len)
        for i, half_band in enumerate(half_band_batch):
            half_band = int(half_band)
            musk_batch[i, max(0, self.fc_index - half_band):self.fc_index + half_band, :] = 1
        musk_batch = musk_batch.view(x.shape[0], x.shape[1], x.shape[3], t_len)  # shape: (batch_size, seq_len, freq_limited, t_len)
        cropped_spectrogram_batch = cropped_spectrogram_batch.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], t_len)
        return cropped_spectrogram_batch, musk_batch

    def snr_band_map(self, snr_batch):
        r5 = snr_batch < 30
        r10 = snr_batch < 40
        r20 = snr_batch < 46
        r30 = snr_batch < 47.5
        half_band_batch = torch.zeros_like(snr_batch)
        half_band_batch[r5] = 0.1 * (snr_batch[r5] - 28) + 20
        half_band_batch[~r5 & r10] = 0.1 * (snr_batch[~r5 & r10] - 33.7) + 35
        half_band_batch[~r10 & r20] = 0.1 * (snr_batch[~r10 & r20] - 42) + 125
        half_band_batch[~r20 & r30] = 0.1 * (snr_batch[~r20 & r30] - 46.5) + 300
        half_band_batch[~r30] = 0.1 * (snr_batch[~r30] - 47.5) + 500
        return half_band_batch.int()


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


class Cropped_Feature(nn.Module):
    def __init__(self, fs, fc, f_low, f_high):
        super().__init__()
        nfft = 8192
        self.spectrogram = Spectrogram(fs, nfft, 16, f_low, f_high)
        self.cropper = Crop(fs, nfft, fc, f_low, f_high)

    def forward(self, x):
        x = self.spectrogram(x)
        x, musk = self.cropper(x)
        return x, musk


class STFT_Magnitude_Feature(Cropped_Feature):
    def __init__(self, fs, fc, f_low, f_high):
        super().__init__(fs, fc, f_low, f_high)

    def forward(self, x):
        x, musk = super().forward(x)
        x = torch.abs(x)
        return x, musk


class STFT_Phase_Feature(Cropped_Feature):
    def __init__(self, fs, fc, f_low, f_high):
        super().__init__(fs, fc, f_low, f_high)

    def forward(self, x):
        x, musk = super().forward(x)
        x = torch.angle(x)
        return x, musk


class CPSD_Phase_Feature(Cropped_Feature):
    def __init__(self, fs, fc, f_low, f_high):
        super().__init__(fs, fc, f_low, f_high)
        self.cpsd_phase = CPSD_Phase_Spectrogram()

    def forward(self, x):
        x, musk = super().forward(x)
        x = self.cpsd_phase(x)
        return x, musk


class CPSD_Phase_Diff_Feature(Cropped_Feature):
    def __init__(self, fs, fc, f_low, f_high):
        super().__init__(fs, fc, f_low, f_high)
        self.cpsd_phase = CPSD_Phase_Spectrogram()

    def forward(self, x):
        x, musk = super().forward(x)
        x = self.cpsd_phase(x)
        x = torch.diff(x, dim=-2)
        return x, musk


FeatureModule = Union[
    Type[STFT_Magnitude_Feature],
    Type[STFT_Phase_Feature],
    Type[CPSD_Phase_Feature],
    Type[CPSD_Phase_Diff_Feature],
]
