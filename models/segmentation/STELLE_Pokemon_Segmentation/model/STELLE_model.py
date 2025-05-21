import torch
import torch.nn as nn


class STELLE_Seg(nn.Module):
    def __init__(self):
        super(STELLE_Seg,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), #H/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # H/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # H/8
        )
        self.middle = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # H/4

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # H/2

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # H
        )

        self.segmentator = nn.Conv2d(16, 14, kernel_size=1)

    def forward(self,x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = self.segmentator(x)
        return x # [B, num_classes, H, W]