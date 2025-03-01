import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import List


def get_gaussian_scoremap(
        one_hot_action: np.ndarray,
        sigma: float=1.0, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :action: with standard deviation :sigma: pixels.
    """
    mu = np.argmax(one_hot_action)
    indices = np.arange(one_hot_action.shape[0])
    scoremap = np.exp(-(indices - mu)**2 / (2 * sigma**2))
    return scoremap

class GomokuDataset(Dataset):
    def __init__(self, npz_file_path, history_length=0):
        print('Loading data from', npz_file_path)

        self.history_length = history_length
        self.episode_data = np.load(npz_file_path, allow_pickle=True)
        
        
        # episode_data:
        #   episode_0:
        #       step_0: (state, action, reward, next_state, done)
        #       step_1: (state, action, reward, next_state, done)
        #       ...
        #   episode_1:
        #       ...

        # reformat data to state action pairs
        self.data = []
        for episode in self.episode_data['replay_buffer']:
            for idx in range(max(0, len(episode) - 2), len(episode)):
                state, action, reward, next_state, done = episode[idx]
                state = state.astype(np.float32)
                next_state = next_state.astype(np.float32)
                # each state and next_state is appended with history board layouts if history_length > 0               
                for i in range(history_length):
                    history_idx = idx - i - 1
                    if history_idx < 0:
                        empty_board = np.zeros((1, state[0].shape[0], state[0].shape[1]), dtype=np.float32)
                        state = np.concatenate((state, empty_board), axis=0)
                    else:
                        history_board_layout = np.expand_dims(episode[history_idx][0][0], axis=0)
                        state = np.concatenate((state, history_board_layout), axis=0)
                # for next_state
                # remove the oldest history board layout from the state and append the next_state
                obs_appends_to_next_state = np.concatenate((np.expand_dims(state[0], axis=0), state[3:-1]), axis=0)
                next_state = np.concatenate((next_state, obs_appends_to_next_state), axis=0)
                # convert action to 2D coordinates
                action_arr = np.zeros(state[0].shape[0]*state[0].shape[1])
                action_arr[action] = 1
                # action_arr = get_gaussian_scoremap(action_arr)
                self.data.append((state, action, reward, next_state, done))        
        self.board_size = self.data[0][0][0][0].shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # state, action, reward, next_state, done
        return self.data[idx]
