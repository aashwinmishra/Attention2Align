"""
Classes for models to be trained.
"""
import torch
import torchvision
import torch.nn as nn


class UNet(nn.Module):
  def contraction_block(self, in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def expansion_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channel, kernel_size, padding=1),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        nn.Conv2d(mid_channel, mid_channel, kernel_size, padding=1),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channel, kernel_size, padding=1),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        nn.Conv2d(mid_channel, out_channels, kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

  def bottleneck_block(self, mid_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=mid_channels, out_channels=2*mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(2*mid_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=2*mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU()
    )

  def __init__(self, in_channels: int=1, out_channels: int=1):
    super().__init__()
    self.conv_encode1 = self.contraction_block(in_channels, out_channels=32)
    self.pool1 = nn.MaxPool2d(2)
    self.conv_encode2 = self.contraction_block(in_channels=32, out_channels=64)
    self.pool2 = nn.MaxPool2d(2)
    self.conv_encode3 = self.contraction_block(in_channels=64, out_channels=128)
    self.pool3 = nn.MaxPool2d(2)

    self.bottleneck = self.bottleneck_block(128)

    self.conv_decode3 = self.expansion_block(256, 128, 64)
    self.conv_decode2 = self.expansion_block(128, 64, 32)
    self.final_layer = self.final_block(64, 32, out_channels)

  def crop_and_concat(self, upsampled, bypass, crop=False):
    if crop:
      c = (bypass.size()[2] - upsampled.size()[2]) // 2
      bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

  def forward(self, x):
    encode_block1 = self.conv_encode1(x)
    encode_pool1 = self.pool1(encode_block1)
    encode_block2 = self.conv_encode2(encode_pool1)
    encode_pool2 = self.pool2(encode_block2)
    encode_block3 = self.conv_encode3(encode_pool2)
    encode_pool3 = self.pool3(encode_block3)
    bottleneck1 = self.bottleneck(encode_pool3)
    decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
    cat_layer2 = self.conv_decode3(decode_block3)
    decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
    cat_layer1 = self.conv_decode2(decode_block2)
    decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
    final_layer = self.final_layer(decode_block1)
    return  final_layer

