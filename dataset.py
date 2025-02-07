import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
  def __init__(self, list_IDs, path, size: int=320):
    self.list_IDs = list_IDs
    self.size = size
    self.path = path
  
  def __len__(self):
    return len(self.list_IDs)
  
  def __getitem__(self, index):
    ID = self.list_IDs[index]
    fixed_image = torch.Tensor(resize(io.imread(self.path + ID), (1, self.size, self.size)))
    moving_image = torch.Tensor(resize(io.imread(self.path + ID), (1, self.size, self.size)))
    return fixed_image, moving_image
