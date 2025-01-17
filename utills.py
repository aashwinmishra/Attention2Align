import numpy as np 
from PIL import Image
import torch 


def affine_grid_np(theta: np.array, size: tuple) -> np.array:
  """
  Numpy implimentation of the affine_grid function in torch.nn.functional.
  Args:
    theta: input batch of Affine Transformation (A_theta) matrices, with 
    shape [N, 2, 3].
    size: tuple of [batch_size, height, width] for the grid.
  Returns:
    Grid of points of shape [batch_size, height, width, 2].
    The last dimension has the x and y co-ordinates of the affine 
    transform of the grid on the target image, that have to be sampled on 
    the source image. 
  """
  batch_size, H, W = size
  x = np.linspace(-1, 1, W)
  y = np.linspace(-1, 1, H)
  xt, yt = np.meshgrid(x, y)
  G = np.stack([xt.flatten(), yt.flatten(), np.ones(H*W)]) #Homogeneous co-ordinates
  G = np.resize(G, (batch_size, 3, H*W)) 
  TG = theta @ G 
  TG = np.reshape(TG, (batch_size, 2, H, W))
  TG = np.moveaxis(TG, 1, -1)
  return TG


def grid_sample_np(input: np.array, grid: np.array) -> np.array:
  """
  Numpy implimentation of the grid_sample function in torch.nn.functional
  Args:
    input: batch of feature maps of shape [batch_size, H_in, W_in, C]
    grid: grid (or "flow field") of shape [batch_size, H_out, W_out, 2]
  Returns:
    Bilinear interpolation based samples of points from the input, 
    of the same dimensions as the grid.
  """
  batch_size, H, W, C = input.shape 
  xs, ys = grid[:, :, :, 0], grid[:, :, :, 1]
  x = (xs + 1.0) * W/2.0 
  y = (ys + 1.0) * H/2.0
  x0 = np.floor(x).astype(np.int64)
  x1 = np.ceil(x).astype(np.int64)
  y0 = np.floor(y).astype(np.int64)
  y1 = np.ceil(y).astype(np.int64)
  x0 = np.clip(x0, 0, W-1)
  x1 = np.clip(x1, 0, W-1)
  y0 = np.clip(y0, 0, H-1)
  y1 = np.clip(y1, 0, H-1)
  Ia = input[np.arange(batch_size)[:,None,None], y0, x0]
  Ib = input[np.arange(batch_size)[:,None,None], y1, x0]
  Ic = input[np.arange(batch_size)[:,None,None], y0, x1]
  Id = input[np.arange(batch_size)[:,None,None], y1, x1]
  wa = (x1-x) * (y1-y)
  wb = (x1-x) * (y-y0)
  wc = (x-x0) * (y1-y)
  wd = (x-x0) * (y-y0)
  wa = np.expand_dims(wa, axis=3)
  wb = np.expand_dims(wb, axis=3)
  wc = np.expand_dims(wc, axis=3)
  wd = np.expand_dims(wd, axis=3)
  return wa*Ia + wb*Ib + wc*Ic + wd*Id

