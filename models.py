import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
