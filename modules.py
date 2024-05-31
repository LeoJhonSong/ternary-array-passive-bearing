import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        return self.conv1d(x)[:, :, :x.size(2)]


class Attention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_dim),
            nn.ReLU(),
            nn.Conv2d(2 * in_dim, 2 * in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_dim),
            nn.ReLU(),
            nn.Conv2d(2 * in_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(AttentionConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.attention = Attention(self.input_dim + self.hidden_dim)
        self.Gates = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding
        )

    def forward(self, x, prev_state):
        prev_h, prev_c = prev_state
        stacked_inputs = torch.cat([x, prev_h], dim=1)  # concatenate along channel axis
        attention = self.attention(stacked_inputs)  # shape: (batch_size, 1, height, width)
        x = attention * x
        stacked_inputs = torch.cat([x, prev_h], dim=1)
        gate_inputs = self.Gates(stacked_inputs)
        forget_gate, input_gate, cell_gate, output_gate = torch.split(gate_inputs, self.hidden_dim, dim=1)
        f = torch.sigmoid(forget_gate)
        i = torch.sigmoid(input_gate)
        c_ = torch.tanh(cell_gate)
        o = torch.sigmoid(output_gate)
        c = f * prev_c + i * c_
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, input_size):
        height, width = input_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Gates.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.Gates.weight.device)
        )


class AttentionConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(AttentionConvLSTMCell(input_dim=cur_input_dim, hidden_dim=self.hidden_dim, kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _, height, width = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :], prev_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
        return layer_output, hidden_state

    def _init_hidden(self, batch_size, input_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, input_size))
        return init_states


class Head(nn.Module):
    def __init__(self, in_channels, label_dim):
        super().__init__()
        head = []
        channels = 512
        for c in in_channels:
            head.append(
                nn.Sequential(
                    nn.Conv2d(c, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    # nn.Linear(channels, label_dim),
                )
            )
        self.head = nn.ModuleList(head)
        self.fc = nn.Linear(channels * len(in_channels), label_dim)

    def forward(self, x1, x2, x3, x4):
        seq_len = x1.shape[1]
        output = []
        for t in range(seq_len):
            output.append(self.fc(torch.cat([
                self.head[0](x1[:, t, :, :, :]),  # shape: (batch_size, channels)
                self.head[1](x2[:, t, :, :, :]),
                self.head[2](x3[:, t, :, :, :]),
                self.head[3](x4[:, t, :, :, :])
            ], dim=1)))
        return torch.stack(output, dim=1)  # shape: (batch_size, seq_len, label_dim)


class Pyramid(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        backbone = backbone.backbone
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class PyramidAttention(nn.Module):
    """输出为以金字塔特征图为通道的通道间注意力后的特征图"""
    def __init__(self, out_channels):
        super().__init__()
        self.upsample = nn.Upsample((128, 8), mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.att1 = Attention(out_channels)
        self.att2 = Attention(out_channels)
        self.att3 = Attention(out_channels)
        self.att4 = Attention(out_channels)

    def forward(self, x1, x2, x3, x4):
        x1 = self.upsample(self.conv1(x1))
        x2 = self.upsample(self.conv2(x2))
        x3 = self.upsample(self.conv3(x3))
        x4 = self.upsample(self.conv4(x4))
        w1 = self.att1(x1)
        w2 = self.att2(x2)
        w3 = self.att3(x3)
        w4 = self.att4(x4)
        x = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4
        return x
