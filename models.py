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


class DecoderBlock(nn.Module):
  """
  Convolutional Block for UNet decoder section. Consisting of 2 consecutive
  convolutional layers, followed by a Transposed Convolution layer.
  Args:
    in_channels: Number of channels in the input to the layer.
    mid_channels: Number of intermediate channels.
    out_channels: Number of channels in the output from the layer. 
  Returns:
    nn.Module sub-class instance.
  """
  def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(mid_channels, out_channels, 
                           kernel_size=3, stride=2, 
                           padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class FinalBlock(nn.Module):
  """
  Convolutional Block for UNet final section. Consisting of 2 consecutive
  convolutional layers.
  Args:
    in_channels: Number of channels in the input to the layer.
    mid_channels: Number of intermediate channels.
    out_channels: Number of channels in the output from the layer. 
  Returns:
    nn.Module sub-class instance.
  """
  def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class UNet(nn.Module):
  """
  Basic UNet model. Takes in batches of pairs of misaligned images and base 
  images. Outputs deformation field of each pair as a 2 channel output.
  Args:
    in_channels: Number of channels in input
    out_channels: Number of channels in output
    features: Number of channels in the encoder and decoder blocks
  Returns:
    nn.Module instance of the UNet.
  """
  def __init__(self, in_channels: int, out_channels: int, features: list):
    super().__init__()
    self.encoder = nn.ModuleList()
    self.decoder = nn.ModuleList()
    self.pool = nn.MaxPool2d(2, 2)

    for num_features in features:
      self.encoder.append(DoubleConv(in_channels, num_features))
      in_channels = num_features

    for num_features in features[::-1]:
      self.decoder.append(DecoderBlock(in_channels, num_features, num_features))
      in_channels = 2*num_features

    self.final_block = FinalBlock(in_channels, in_channels, 2)

  def forward(self, x):
    skip_connections = []
    for block in self.encoder:
      x = block(x)
      skip_connections.append(x)
      x = self.pool(x)
    skip_connections = skip_connections[::-1]
    
    for i in range(len(self.decoder)):
      x = self.decoder[i](x)
      x = torch.cat([skip_connections[i], x], dim=1)
    x = self.final_block(x)

    return x


class U_Net(nn.Module):
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
    self.final_section = nn.Conv2d(features[0], out_channels, kernel_size=1)

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


class AttentionAlign(nn.Module):
  def __init__(self, 
               ae_type: str='UNet',
               in_channels: int=2, 
               out_channels: int=2, 
               features: list=[16, 32, 32, 32],
               device: torch.device=torch.device('cpu')) -> None:
    super().__init__()
    if ae_type is 'UNet':
      self.autoencoder = UNet(in_channels, out_channels, features)
    # elif ae_type is 'LinkNet':
    #   self.autoencoder = LinkNet(in_channels, out_channels, features)
    # else:
    #   self.autoencoder = AutoEncoder(in_channels, out_channels, features)

