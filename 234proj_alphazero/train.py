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


if __name__ == "__main__":
    
    config = AlphaZeroConfig()
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    # main body of training
    run_selfplay(config, storage, replay_buffer)
