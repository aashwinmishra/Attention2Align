import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
  """
  Basic Convolutional Block for AutoEncoders. Consisting of 2 consecutive
  convolutional layers.
  Args:
    in_channels: Number of channels in the input to the layer.
    out_channels: Number of channels in the output from the layer. 
  Returns:
    nn.Module sub-class instance.
  """
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class UNet(nn.Module):
  """
  Classic UNet from Ronneberger et al (2015).
  Args:
    in_channels: Number of channels in input
    out_channels: Number of channels in output
    features: Number of channels in the encoder and decoder blocks
  Returns:
    nn.Module instance of the classic UNet.
  """
  def __init__(self, 
               in_channels: int = 2,
               out_channels: int = 2,
               features = [64, 128, 256, 512]
               ):
    super().__init__()
    self.encoder = nn.ModuleList()
    self.decoder = nn.ModuleList()
    self.pool = nn.MaxPool2d(2, 2)

    for feature in features:
      self.encoder.append(DoubleConv(in_channels, feature))
      in_channels = feature
      
    for feature in features[::-1]:
      self.decoder.append(nn.ConvTranspose2d(2*feature, feature, 
                                             kernel_size=2, 
                                             stride=2))
      self.decoder.append(DoubleConv(2*feature, feature))

    self.bottleneck = DoubleConv(features[-1], 2*features[-1])
    self.final_section = DoubleConv(features[0], out_channels)

  def forward(self, x):
    skip_connections = []
    for block in self.encoder:
      x = block(x)
      skip_connections.append(x)
      x = self.pool(x)
    x = self.bottleneck(x)
    skip_connections = skip_connections[::-1]
    for idx in range(0, len(self.decoder), 2):
      x = self.decoder[idx](x)
      skip_connection = skip_connections[idx//2]
      concat_skip = torch.cat((skip_connection, x), dim=1)
      x = self.decoder[idx+1](concat_skip)
    return self.final_section(x)

