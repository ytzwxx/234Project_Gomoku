import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import List

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
            for idx in range(len(episode)):
                state, action, reward, next_state, done = episode[idx]
                # each state and next_state is appended with history board layouts if history_length > 0
                if type(state) == list or List:
                    state = np.array(state)
                    next_state = np.array(next_state)
                
                for i in range(history_length):
                    history_idx = idx - i - 1
                    if history_idx < 0:
                        empty_board = np.zeros((1, state[0].shape[0], state[0].shape[1]))
                        state = np.concatenate((state, empty_board), axis=0)
                    else:
                        history_board_layout = np.expand_dims(episode[history_idx][0][0], axis=0)
                        state = np.concatenate((state, history_board_layout), axis=0)
                # for next_state
                # remove the oldest history board layout from the state and append the next_state
                obs_appends_to_next_state = np.concatenate((np.expand_dims(state[0], axis=0), state[3:-1]), axis=0)
                next_state = np.concatenate((next_state, obs_appends_to_next_state), axis=0)
                self.data.append((state, action, reward, next_state, done))        
        self.board_size = self.data[0][0][0][0].shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # state, action, reward, next_state, done
        return self.data[idx]
