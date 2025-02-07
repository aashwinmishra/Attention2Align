import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
from losses import mse_loss
from models import AttentionAlign

class Attention2Align():
    def __init__(self, input_dims):
        self.dims = input_dims
        self.model = AttentionAlign
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.99)
        self.params = {'batch_size': 25,'shuffle': True}
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
      
    def forward(self, x):
        self.check_dims(x)
        return self.model(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = mse_loss(y, ytrue, n, lamda)
        return loss

    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, return_metric_score=True):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        registered_image = self.model(batch_moving, batch_fixed)
        train_loss = self.calculate_loss(
            registered_image, batch_fixed, n, lamda)
        train_loss.backward()
        self.optimizer.step()
        return train_loss

    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            registered_image = self.model(batch_moving, batch_fixed)
            val_loss = mse_loss(registered_image, batch_fixed)
            return val_loss


