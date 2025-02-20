import torch
import torch.nn as nn 
from torch.nn import functional as F


def TV(img: torch.tensor) -> float:
  """
  Computes the Total Variation Metric of a given image,
  the l_1 norm is used for simplicity.
  Args:
    img: input image as a tensor
  Returns:
    float value of the metric.
  """
  if img.ndim != 2:
    raise RuntimeError(f"Expected input `img` to be an 2D tensor, but got {img.shape}")
  diff1 = img[1:, :] - img[:-1, :]
  diff2 = img[:, 1:] - img[:, :-1]
  res1 = diff1.abs().sum()
  res2 = diff2.abs().sum()
  score = res1 + res2
  return score


def DoM(img: torch.tensor) -> float:
  """
  Computes the Difference Of differences in a Median filtered image 
  Metric of a given image.
  Args:
    img: input image as a tensor
  Returns:
    float value of the metric.
  """  
  diffx = img[2:, :] + img[:2, :] - 2 * img[:, :]
  pass


