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
from tqdm import tqdm

if __name__ == "__main__":
    
    config = AlphaZeroConfig()
    storage = SharedStorage(config)
    replay_buffer = ReplayBuffer(config)
    
    # main body of training
    print("Training begins.")
    for self_play_round in tqdm(range(config.max_iter), desc="Self play round:"):
        for actor in range(config.num_actors): # samples. Convert from parrallel to sequential
            print(f"Self-play round {self_play_round} actor {actor} begin.")
            # run_selfplay(config, storage, replay_buffer)
            threads = []
            for i in range(config.parrallel_actor):
                thread = threading.Thread(target=run_selfplay, args=(config, storage, replay_buffer))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            

        print(f"Self-play round {self_play_round} complete.")
        storage.self_play_round = self_play_round
        train_network(config, storage, replay_buffer)

        print(f"Training round {self_play_round} losses \n \
                value loss: {storage.value_loss[-1]} \n \
                policy loss: {storage.policy_loss[-1]} \n \
                total loss: {storage.total_loss[-1]}")

        if self_play_round % config.save_play_time_interval == 0 and config.save_after_each_play_interval:
            storage.save_network_after_play()
    
    storage.save_loss()
    print("Training complete.")