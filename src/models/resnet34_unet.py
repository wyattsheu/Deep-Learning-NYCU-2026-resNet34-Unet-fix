import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34_UNet, self).__init__()

        # --- Encoder: ResNet34 從零實作 ---
        self.inplanes = 64
        # 初始層 (Stem)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Stages
        self.layer1 = self._make_layer(BasicBlock, 64, 3)  # Stage 1
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # Stage 2
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  # Stage 3
        self.layer4 = self._make_layer(
            BasicBlock, 512, 3, stride=2
        )  # Stage 4 (Bottleneck)

        # --- Decoder: UNet 結構 ---
        # 注意：ResNet 的輸出通道分別是 64, 128, 256, 512
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)  # 256 (up) + 256 (skip)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 (up) + 128 (skip)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # 64 (up) + 64 (skip)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)  # 64 (up) + 64 (skip)

        # 補回最初的 7x7 conv 造成的解析度下降
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder Path
        x0 = self.relu(self.bn1(self.conv1(x)))  # 1/2 size (112x112)
        s1 = self.layer1(self.maxpool(x0))  # 1/4 size (56x56)
        s2 = self.layer2(s1)  # 1/8 size (28x28)
        s3 = self.layer3(s2)  # 1/16 size (14x14)
        s4 = self.layer4(s3)  # 1/32 size (7x7)

        # Decoder Path
        x = self.up4(s4)
        x = torch.cat([x, s3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x0], dim=1)  # 與初始層進行 Concat
        x = self.dec1(x)

        x = self.final_up(x)
        return self.final_conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DoubleConv(nn.Module):
    """這裡使用 Padding=1 確保解析度不會像你原本的代碼那樣一直縮小，
    這樣在結合 ResNet 時會比較好對齊。"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # 用於維度不匹配時的 1x1 Conv
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 殘差相加
        out = self.relu(out)

        return out
