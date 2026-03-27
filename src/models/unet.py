import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        """
        TODO: 從零開始 (From scratch) 搭建 UNet 的 Encoder, Bottleneck, 與 Decoder。
        不可載入任何預訓練權重。
        """
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        self._initialize_weights()

    @staticmethod
    def _center_crop(skip, target):
        _, _, h, w = skip.shape
        _, _, th, tw = target.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return skip[:, :, dh : dh + th, dw : dw + tw]

    def forward(self, x):
        # Paper-style U-Net uses valid conv, so skip features are center-cropped before concat.
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        x = self.bottleneck(self.pool(s4))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, self._center_crop(s4, x)], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, self._center_crop(s3, x)], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, self._center_crop(s2, x)], dim=1))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, self._center_crop(s1, x)], dim=1))

        return self.output(x)

    def _initialize_weights(self):
        """實作論文中提到的 Kaiming Normal (He Initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
