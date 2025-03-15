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

    self_play_time = np.load('234proj_alphazero/training_results_trial_3/self_play_round_9/self_play_times.npy')



    play_iter = np.arange(0, len(self_play_time))
    # plt.plot(play_iter, play_round_policy_loss, label='Play Round Policy Loss')
    plt.plot(play_iter, self_play_time, label='Self-play time')
    # plt.plot(play_iter, play_round_total_loss, label='Play Round Total Loss')
    
    # plt.plot(test_epochs, test_loss, label='Test Loss')

    plt.xlabel('Train iterations')
    plt.ylabel('Self-play time')
    plt.legend()
    # save as png
    # plt.savefig('play_round_losses.png')
    plt.savefig('self-play_time.png')
    mean_time = np.mean(self_play_time)
    print(f"Mean self-play time: {mean_time}")
    # plt.show()
