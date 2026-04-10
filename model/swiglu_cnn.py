"""CNN with SwiGLU activation for comparison with InnerNet CNN.

SwiGLUCNN replaces ReLU with SwiGLU gating: Swish(conv_a) ⊙ conv_b.
Uses same channel structure as InnerNet CNN (2× channels then halve via gating).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwiGLUCNN(nn.Module):
    """CNN with SwiGLU activation for CIFAR/MNIST.

    Same architecture as paper's CNN but replaces ReLU with SwiGLU:
    Conv2d → BatchNorm → SwiGLU(split channels) → MaxPool → Dropout

    out_channel specifies the OUTPUT channels (after gating halves).
    Internal conv uses 2× channels.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        if config.dataset.name in ('mnist', 'fashionmnist'):
            x_size = 28
        elif config.dataset.name in ('cifar10', 'cifar100', 'svhn', 'stl10'):
            x_size = 32
        else:
            raise ValueError("Non-supported dataset!")

        layers = []
        in_ch = self.input_channel
        for i in range(len(self.out_channel)):
            out_ch = self.out_channel[i]
            # Conv outputs 2× channels, SwiGLU halves them
            layers.append(nn.Conv2d(in_ch, out_ch * 2, kernel_size=self.kernel_size[i],
                                     stride=self.stride[i], padding=self.zero_pad[i]))
            x_size = (x_size - self.kernel_size[i] + 2 * self.zero_pad[i]) // self.stride[i] + 1
            layers.append(nn.BatchNorm2d(out_ch * 2))
            layers.append(SwiGLUActivation2d())
            if x_size >= 2:
                layers.append(nn.MaxPool2d(kernel_size=2))
                x_size = x_size // 2
            layers.append(nn.Dropout(p=self.dropout))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.fc_out = nn.Linear(self.out_channel[-1] * x_size * x_size, self.num_classes)

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = nn.MSELoss()

        self._init_param()

    def _init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class SwiGLUActivation2d(nn.Module):
    """SwiGLU activation for conv feature maps.

    Input: (B, 2C, H, W) → split into gate (B, C, H, W) and value (B, C, H, W)
    Output: swish(gate) * value → (B, C, H, W)
    """
    def forward(self, x):
        C = x.size(1) // 2
        gate, value = x[:, :C], x[:, C:]
        return F.silu(gate) * value
