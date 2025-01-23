import torch
import torch.nn as nn 
from torch.nn import functional as F


def mse_loss(y_pred: torch.tensor, y_target: torch.tensor) -> torch.tensor:
  """
  Takes 2 batches of image tensor, of shapes 
  [batch_size, channels, height, width]. Returns pixel wise mse loss.
  Args:
    y_pred: Prediction from spatial transformer network
    y_target: baseline image batch
  Returns:
    tensor value of the loss.
  """
  return F.mse_loss(y_pred, y_target)


def cc_loss(y_pred: torch.tensor, y_target: torch.tensor) -> torch.tensor:
  """
  Takes 2 batches of image tensor, of shapes 
  [batch_size, channels, height, width]. Returns normalized cross-correlation 
  between the 2.
  Args:
    y_pred: Prediction from spatial transformer network
    y_target: baseline image batch
  Returns:
    tensor value of the loss.
  """
  return F.mse_loss(y_pred, y_target)
