import os
import glob
import torch
import pickle
import numpy as np
from collections import defaultdict
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal

from torch.utils.data import Dataset
from utils.data_helper import *

__all__ = ['InnerNetData']


class InnerNetData(Dataset):

  def __init__(self, config, split='train'):
    assert split in ['train', 'val', 'test'], "no such split"
    self.config = config
    self.split = split
    self.npr = np.random.RandomState(seed=config.seed)

    arg_in_dim = getattr(config.model, 'arg_in_dim', 2)

    if arg_in_dim == 1:
      # 1D: smoothed random function on a line
      nb = 101
      x = np.linspace(-5, 5, nb)
      xy = x.reshape(-1, 1)
      from scipy.stats import norm
      gaussian_kernel = norm.pdf(x, loc=0, scale=1/3)
      gaussian_kernel /= gaussian_kernel.sum()
      init_unif = self.npr.uniform(-1, 1, size=nb)
      targets = np.convolve(init_unif, gaussian_kernel, mode='same').reshape(-1, 1)
    else:
      # 2D: smoothed random function on a grid (original)
      nb = 101
      x = np.linspace(-5, 5, nb)
      y = np.linspace(-5, 5, nb)
      xv, yv = np.meshgrid(x, y)
      xy = np.vstack([xv.reshape(-1), yv.reshape(-1)]).T
      mvn = multivariate_normal(mean=[0, 0], cov=[[1/9, 0], [0, 1/9]])
      gaussian_kernel = mvn.pdf(xy).reshape(nb, nb)
      gaussian_kernel /= gaussian_kernel.sum()
      init_unif = self.npr.uniform(-1, 1, size=(nb, nb))
      targets = convolve2d(init_unif, gaussian_kernel, mode='same').reshape(-1, 1)

    self.xy = xy
    self.targets = targets

  def __getitem__(self, index):
    return self.xy[index, None], self.targets[index, None]

  def __len__(self):
    return len(self.xy)

  def collate_fn(self, batch):
    assert isinstance(batch, list)

    xy_batch = torch.from_numpy(np.concatenate([bch for bch, _ in batch], axis=0)).float()
    targets_batch = torch.from_numpy(np.concatenate([bch for _, bch in batch], axis=0)).float()

    return xy_batch, targets_batch