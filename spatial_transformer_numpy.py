import numpy as np
from PIL import Image


def affine_grid(theta: np.array, size: tuple) -> np.array:
  """
  Numpy implimentation of the affine_grid function in torch.nn.functional.
  Limited to 3 channel images.
  Args:
    theta: input batch of affine matrices with shape [N, 2, 3].
    size: tuple of [batch_size, height, width] for the grid.
  Returns:
    Grid of points of shape [batch_size, height, width, 2].
    The last dimension has the x and y co-ordinates of the affine 
    transform of the grid on the target image, that have to be sampled on 
    the source image. 
  """
  batch_size = theta.shape[0]
  H, W = size[0], size[1]
  x = np.linspace(-1, 1, W)
  y = np.linspace(-1, 1, H)
  x_t, y_t = np.meshgrid(x, y)
  sampling_grid = np.stack([x_t.flatten(), y_t.flatten(), np.ones_like(x_t.flatten())])
  sampling_grid = np.resize(sampling_grid, (batch_size, 3, H*W))  
  batch_grids = M @ sampling_grid 
  batch_grids = batch_grids.reshape(batch_size, 2, H, W)
  batch_grids = np.moveaxis(batch_grids, 1, -1)
  return batch_grids


def grid_sample(input, grid):
  """
  Numpy implimentation of the grid_sample function in torch.nn.functional
  Args:
    input: batch of feature maps of shape [N, H_in, W_in, 3]
    grid: grid (or "flow field") of shape [N, H_out, W_out, 2]
  Returns:
    Bilinear samples of points from the input, of the same dimensions as the grid
  """
  batch_size, H, W, C = input.shape
  xs, ys = grid[:,:,:,0], grid[:,:,:,1]
  x = (xs + 1.0) * W/2.0
  y = (ys + 1.0) * H/2.0
  x0 = np.floor(x).astype(np.int64)
  x1 = x0 + 1
  y0 = np.floor(y).astype(np.int64)
  y1 = y0 + 1
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

