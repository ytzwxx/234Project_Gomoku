import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import data_loader, networks, gym_utils
from utils.networks import np2torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_loss = np.load('initial_training/train_losses.npy')
    test_loss = np.load('initial_training/test_losses_every_10_epoches.npy')

    test_epochs = np.arange(1, len(test_loss)+1) * 10
    train_epochs = np.arange(1, len(train_loss)+1)

    plt.plot(train_epochs, train_loss, label='Train Loss')
    # plt.plot(test_epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
