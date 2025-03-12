from network import *
from config import AlphaZeroConfig
import copy
import os
import torch
import matplotlib.pyplot as plt
# TODO
def make_uniform_network():
    uniform_network = AlphaZeroNet()
    return uniform_network

class SharedStorage(object):

  def __init__(self, config: AlphaZeroConfig):
    self._networks = {}
    self._base_idx = 0
    self.config = config
    self.save_folder = config.save_folder
    self.self_play_round = 0
    
    self.value_loss = []
    self.policy_loss = []
    self.total_loss = []
  
  def latest_network(self) -> AlphaZeroNet:
    if self._networks:
      return copy.deepcopy(self._networks[max(self._networks.keys())].to(device))
    else:
      return make_uniform_network().to(device)

  def save_network(self, step: int, network: AlphaZeroNet):
    self._networks[step] = network # Caution: this is overwriting the network at step if it already exists
  
    if self.config.save_after_each_checkpoint_interval:
      # save network to disk    
      os.makedirs(os.path.join(self.save_folder, f'self_play_round_{self.self_play_round}'), exist_ok=True)
      torch.save(network.state_dict(), f'{self.save_folder}/self_play_round_{self.self_play_round}/network_training_step_{step}.pth')

  def get_num_networks(self):
    return len(self._networks)

  def update_base_idx(self, new_base_idx):
    self._base_idx += new_base_idx
  
  def save_network_after_play(self):
    torch.save(self.latest_network().state_dict(), f'{self.save_folder}/self_play_round_{self.self_play_round}/latest_network.pth')

  def record_loss(self, value_loss, policy_loss, total_loss):
    # index is total number of training steps = config.training_steps * config.max_iter
    self.value_loss.append(value_loss)
    self.policy_loss.append(policy_loss)
    self.total_loss.append(total_loss)

  def save_loss(self):
    np.save(f'{self.save_folder}/value_loss.npy', np.array(self.value_loss))
    np.save(f'{self.save_folder}/policy_loss.npy', np.array(self.policy_loss))
    np.save(f'{self.save_folder}/total_loss.npy', np.array(self.total_loss))