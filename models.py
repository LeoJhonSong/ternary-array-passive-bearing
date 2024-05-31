from typing import Literal, Type

import torch
from torch import nn
from torchvision.models import resnet18, vit_b_32

import feature_modules
import modules


class Custom_ResNet18(nn.Module):
    def __init__(self, feature_module: Type[feature_modules.STFT_Magnitude_Feature | feature_modules.CPSD_Phase_Feature | feature_modules.CPSD_Phase_Diff_Feature], fs, fc, f_low, f_high, label_type: Literal['direction', 'position']):
        super().__init__()
        self.feature = feature_module(fs, fc, f_low, f_high)
        self.backbone = resnet18(num_classes=2 if label_type == 'direction' else 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = x.squeeze(1)
        x = self.backbone(x)
        return x


class Custom_ViT_B_32(nn.Module):
    def __init__(self, feature_module: feature_modules.STFT_Magnitude_Feature | feature_modules.CPSD_Phase_Feature | feature_modules.CPSD_Phase_Diff_Feature, fs, fc, f_low, f_high, label_type: Literal['direction', 'position']):
        super().__init__()
        self.feature = feature_module(fs, fc, f_low, f_high)
        self.backbone = vit_b_32(num_classes=2 if label_type == 'direction' else 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = x.squeeze(1)
        x = self.backbone(x)
        return x


class CPSD_Phase_Diff_Resnet18_attConvLSTM(nn.Module):
    def __init__(self, fs, fc, f_low, f_high, label_type: Literal['direction', 'position'], backbone: Custom_ResNet18):
        super().__init__()
        self.feature = feature_modules.CPSD_Phase_Diff_Feature(fs, fc, f_low, f_high)
        self.pyramid = modules.Pyramid(backbone)
        self.attConvLSTM1 = modules.AttentionConvLSTM(64, 64, (3, 3), 1)  # TODO: 试试hidden_dim更小会怎样
        self.attConvLSTM2 = modules.AttentionConvLSTM(128, 128, (3, 3), 1)
        self.attConvLSTM3 = modules.AttentionConvLSTM(256, 256, (3, 3), 1)
        self.attConvLSTM4 = modules.AttentionConvLSTM(512, 512, (3, 3), 1)
        self.head = modules.Head((64, 128, 256, 512), 2 if label_type == 'direction' else 3)

        # freeze the backbone
        for param in self.pyramid.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        seq_len = x.shape[1]
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # shape: (batch_size * seq_len, channels, h, w)
        x1, x2, x3, x4 = self.pyramid(x)
        x1, x2, x3, x4 = self.to_seq([x1, x2, x3, x4], seq_len)  # shape: (batch_size, seq_len, channels, h, w)
        x1, _ = self.attConvLSTM1(x1)  # shape: (batch_size, seq_len, 64, 482, 8)
        x2, _ = self.attConvLSTM2(x2)  # shape: (batch_size, seq_len, 128, 241, 4)
        x3, _ = self.attConvLSTM3(x3)  # shape: (batch_size, seq_len, 256, 121, 2)
        x4, _ = self.attConvLSTM4(x4)  # shape: (batch_size, seq_len, 512, 61, 1)
        x = self.head(x1, x2, x3, x4)  # shape: (batch_size, seq_len, label_dim)
        return x.contiguous()

    def to_seq(self, tensors, seq_len):
        for i, tensor in enumerate(tensors):
            tensor = tensor.view(-1, seq_len, tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])  # shape: (batch_size, seq_len, channels, h, w)
            tensors[i] = tensor
        return tensors
