from train_utils import *
from config import AlphaZeroConfig
from game import GomokuGame, Node
from replay_buffer import ReplayBuffer
from sharedstorage import SharedStorage
from network import *
import math
from typing import List
import random
import numpy as np
import threading

if __name__ == "__main__":
    
    config = AlphaZeroConfig()
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    # main body of training
    for _ in range(config.max_iter):
        for _ in range(config.num_actors):
            run_selfplay(config, storage, replay_buffer)
        train_network(config, storage, replay_buffer)