"""ResNet models with optional InnerNet activation for CIFAR-10.

Tests whether skip connections make InnerNet redundant.
BaselineResNet uses standard ReLU; InnerNetResNet replaces ReLU with learned 2-arg activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet basic block with configurable activation."""
    def __init__(self, in_ch, out_ch, stride=1, act_fn=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = act_fn if act_fn else nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class InnerNetAct(nn.Module):
    """2-arg InnerNet activation for use inside ResNet blocks.

    Takes a feature map (B, C, H, W), pairs adjacent channels,
    applies a small MLP (2→hidden→1) per pair, producing (B, C//2, H, W).
    To keep channel count, we double the conv output channels.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pair adjacent channels: (B, C, H, W) → (B, C//2, 2, H, W)
        x = x.view(B, C // 2, 2, H, W)
        # Reshape for linear: (B*C//2*H*W, 2)
        x = x.permute(0, 1, 3, 4, 2).reshape(-1, 2)
        x = self.net(x)
        # Back to (B, C//2, H, W)
        x = x.view(B, C // 2, H, W)
        return x


class InnerNetBlock(nn.Module):
    """ResNet block with InnerNet activation.

    Conv outputs 2× channels, InnerNet halves them back.
    """
    def __init__(self, in_ch, out_ch, stride=1, inner_hidden=32):
        super().__init__()
        # Double output channels so InnerNet can pair them
        self.conv1 = nn.Conv2d(in_ch, out_ch * 2, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch * 2)
        self.inner1 = InnerNetAct(inner_hidden)

        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch * 2)
        self.inner2 = InnerNetAct(inner_hidden)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.inner1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.inner2(out)
        out += self.shortcut(x)
        return out


class InnerNetInternalBlock(nn.Module):
    """ResNet block with InnerNet ONLY at the internal position (between conv1 and conv2).

    Post-skip activation stays as ReLU. This mirrors the Transformer FFN
    approach where InnerNet replaces activation inside the FFN but residual
    connection wraps outside.
    """
    def __init__(self, in_ch, out_ch, stride=1, inner_hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch * 2, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch * 2)
        self.inner = InnerNetAct(inner_hidden)  # only internal activation is InnerNet

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.inner(self.bn1(self.conv1(x)))   # InnerNet (no skip protection)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)                            # ReLU after skip (standard)
        return out


class InnerNetInternalResNet(nn.Module):
    """ResNet-18 with InnerNet only at internal positions (between conv1 and conv2).
    Post-skip activation remains ReLU."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes
        inner_hidden = getattr(config.model, 'inner_hidden', 32)

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 2, stride=1, inner_hidden=inner_hidden)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, inner_hidden=inner_hidden)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, inner_hidden=inner_hidden)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, inner_hidden=inner_hidden)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, inner_hidden):
        layers = [InnerNetInternalBlock(in_ch, out_ch, stride, inner_hidden)]
        for _ in range(1, num_blocks):
            layers.append(InnerNetInternalBlock(out_ch, out_ch, 1, inner_hidden))
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class BaselineResNet(nn.Module):
    """ResNet-18 style for CIFAR-10 with ReLU."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = nn.MSELoss()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        loss = self.loss_func(out, labels)
        return out, loss, []


class InnerNetResNet(nn.Module):
    """ResNet-18 style for CIFAR-10 with InnerNet activation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_classes = config.model.num_classes
        inner_hidden = getattr(config.model, 'inner_hidden', 32)

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # First layer uses ReLU

        self.layer1 = self._make_layer(64, 64, 2, stride=1, inner_hidden=inner_hidden)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, inner_hidden=inner_hidden)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, inner_hidden=inner_hidden)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, inner_hidden=inner_hidden)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        if config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = nn.MSELoss()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, inner_hidden=32):
        layers = [InnerNetBlock(in_ch, out_ch, stride, inner_hidden)]
        for _ in range(1, num_blocks):
            layers.append(InnerNetBlock(out_ch, out_ch, 1, inner_hidden))
        return nn.Sequential(*layers)

    def forward(self, x, labels, collect=False):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        loss = self.loss_func(out, labels)
        return out, loss, []
