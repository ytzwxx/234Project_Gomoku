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

    # train_loss = np.load('initial_234proj_alphazero/training_results/train_losses.npy')
    # test_loss = np.load('initial_234proj_alphazero/training_results/test_losses_every_10_epoches.npy')

    # test_epochs = np.arange(1, len(test_loss)+1) * 10
    # train_epochs = np.arange(1, len(train_loss)+1)

    policy_loss = np.load('234proj_alphazero/training_results/policy_loss.npy')
    value_loss = np.load('234proj_alphazero/training_results/value_loss.npy')
    total_loss = np.load('234proj_alphazero/training_results/total_loss.npy')

    play_round_policy_loss = np.load('234proj_alphazero/training_results/play_round_policy_loss.npy')
    play_round_value_loss = np.load('234proj_alphazero/training_results/play_round_value_loss.npy')
    play_round_total_loss = np.load('234proj_alphazero/training_results/play_round_total_loss.npy')

    # train_iters = np.arange(1, len(policy_loss)+1)
    # plt.plot(train_iters, policy_loss, label='Policy Loss')
    # plt.plot(train_iters, value_loss, label='Value Loss')
    # plt.plot(train_iters, total_loss, label='Total Loss')

    play_iter = np.arange(0, 5 * len(play_round_policy_loss), 5) 
    plt.plot(play_iter, play_round_policy_loss, label='Play Round Policy Loss')
    plt.plot(play_iter, play_round_value_loss, label='Play Round Value Loss')
    plt.plot(play_iter, play_round_total_loss, label='Play Round Total Loss')
    
    # plt.plot(test_epochs, test_loss, label='Test Loss')

    plt.xlabel('Train iterations')
    plt.ylabel('Loss')
    plt.legend()
    # save as png
    plt.savefig('play_round_losses.png')
        # plt.show()
