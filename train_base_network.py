import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import data_loader, networks, gym_utils

if __name__ == '__main__':
    npz_file_path = 'data/gomoku_trajectories.npz'

    cfg = {
        'board_size': 8,
        'history_length': 4,
    }
    dataset = data_loader.GomokuDataset(npz_file_path, history_length=cfg['history_length'])

    dataset[2]

