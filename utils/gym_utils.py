import gym
from gym import spaces
import numpy as np
import random

def action2activation(action, board_size):
    """
    Convert action to 2-D activation map.
    action: int, represents 1-D index on board
    board_size: int, size of the board
    Return:
        2-D activation map with shape (board_size, board_size)
    """
    row = action // board_size
    col = action % board_size
    activation = np.zeros((board_size, board_size))
    activation[row, col] = 1
    return activation

def gymboard2np(board: spaces.Box):
    """
    Convert gym board to numpy array.
    board: spaces.Box, Current state on the board
    Return:
        2-D numpy array
    """
    return np.array(board)