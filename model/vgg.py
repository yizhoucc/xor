"""VGG-16 with BatchNorm for CIFAR-10/100.

Adapted for 32×32 images (smaller FC layers than ImageNet version).
"""
import torch
import torch.nn as nn


class VGG16BN(nn.Module):
    """VGG-16+BN for CIFAR. Standard baseline architecture."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes

        self.features = self._make_layers([
            64, 64, 'M',
            128, 128, 'M',
            256, 256, 256, 'M',
            512, 512, 512, 'M',
            512, 512, 512, 'M',
        ])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class InnerNetVGGActivation(nn.Module):
    """InnerNet activation for VGG: pairs adjacent channels."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        # x: (B, 2C, H, W) → pair → (B, C, H, W)
        B, C2, H, W = x.shape
        C = C2 // 2
        x = x.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2).reshape(-1, 2)
        return self.net(x).view(B, C, H, W)


class InnerNetVGG16(nn.Module):
    """VGG-16 with InnerNet activation. Conv outputs 2× channels, InnerNet halves."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes
        inner_hidden = getattr(config.model, 'inner_hidden', 32)

        self.inner_net = InnerNetVGGActivation(inner_hidden)

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
               512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes))

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # 2× channels → InnerNet halves
                layers.extend([
                    nn.Conv2d(in_channels, v * 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v * 2),
                    self.inner_net,
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class SwiGLUVGGActivation(nn.Module):
    """SwiGLU activation for VGG: SiLU(gate) * value on paired channels."""
    def forward(self, x):
        C = x.size(1) // 2
        gate, value = x[:, :C], x[:, C:]
        return torch.nn.functional.silu(gate) * value


class SwiGLUVGG16(nn.Module):
    """VGG-16 with SwiGLU activation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
               512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes))

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        swiglu = SwiGLUVGGActivation()
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v * 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v * 2),
                    swiglu,
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class WideResNetBlock(nn.Module):
    """Wide residual block."""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + self.shortcut(x)


class InnerNetInternalWRNBlock(nn.Module):
    """Wide residual block with InnerNet only at the internal position.

    inner_net is shared across all blocks.
    """
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3, inner_net=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch * 2, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch * 2)
        self.inner = inner_net  # shared
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))   # position 1: ReLU (post-skip, keep)
        out = self.dropout(out)
        out = self.inner(self.bn2(out))              # position 2: InnerNet (internal)
        out = self.conv2(out)
        return out + self.shortcut(x)


class InnerNetInternalWRN(nn.Module):
    """WideResNet-28-10 with InnerNet only at internal positions (shared)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes
        depth = getattr(config.model, 'depth', 28)
        widen = getattr(config.model, 'widen_factor', 10)
        dropout = getattr(config.model, 'dropout', 0.3)
        inner_hidden = getattr(config.model, 'inner_hidden', 32)

        self.inner_net = InnerNetVGGActivation(inner_hidden)  # shared

        n = (depth - 4) // 6
        channels = [16, 16 * widen, 32 * widen, 64 * widen]

        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)

        self.group1 = self._make_group(channels[0], channels[1], n, stride=1, dropout=dropout)
        self.group2 = self._make_group(channels[1], channels[2], n, stride=2, dropout=dropout)
        self.group3 = self._make_group(channels[2], channels[3], n, stride=2, dropout=dropout)

        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

    def _make_group(self, in_ch, out_ch, n_blocks, stride, dropout):
        layers = [InnerNetInternalWRNBlock(in_ch, out_ch, stride, dropout, inner_net=self.inner_net)]
        for _ in range(1, n_blocks):
            layers.append(InnerNetInternalWRNBlock(out_ch, out_ch, 1, dropout, inner_net=self.inner_net))
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = torch.relu(self.bn(out))
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class WideResNet(nn.Module):
    """WideResNet-28-10 for CIFAR. Strong baseline (~80-82% on CIFAR-100)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes
        depth = getattr(config.model, 'depth', 28)
        widen = getattr(config.model, 'widen_factor', 10)
        dropout = getattr(config.model, 'dropout', 0.3)

        n = (depth - 4) // 6  # blocks per group
        channels = [16, 16 * widen, 32 * widen, 64 * widen]

        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=1, bias=False)

        self.group1 = self._make_group(channels[0], channels[1], n, stride=1, dropout=dropout)
        self.group2 = self._make_group(channels[1], channels[2], n, stride=2, dropout=dropout)
        self.group3 = self._make_group(channels[2], channels[3], n, stride=2, dropout=dropout)

        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

    def _make_group(self, in_ch, out_ch, n_blocks, stride, dropout):
        layers = [WideResNetBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n_blocks):
            layers.append(WideResNetBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = torch.relu(self.bn(out))
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        loss = self.loss_func(out, labels)
        return out, loss, []
