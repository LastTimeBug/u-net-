import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 定义解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # 输出通道数为1，表示二分类问题
        )

    def forward(self, x):
        # 编码器部分
        encoder_output1 = self.encoder[0:4](x)  # 第一层到第四层
        encoder_output2 = self.encoder[4:8](encoder_output1)  # 第五层到第八层
        encoder_output3 = self.encoder[8:](encoder_output2)  # 第九层到最后一层
        # 解码器部分
        decoder_output = self.decoder(encoder_output3)
        return torch.sigmoid(decoder_output)  # 使用 Sigmoid 激活函数，将输出限制在 0 到 1 之间