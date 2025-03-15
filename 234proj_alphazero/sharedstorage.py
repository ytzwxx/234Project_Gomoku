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

    self.play_round_value_loss = []
    self.play_round_policy_loss = []
    self.play_round_total_loss = []

    self.this_round_value_loss = []
    self.this_round_policy_loss = []
    self.this_round_total_loss = []

    self.use_history_network = False
    self.history_network = AlphaZeroNet()

    self.self_play_times = []

    if config.use_history_network:
      self.load_network(config.history_network_path)
  
  def load_network(self, path):
    print(f'Loading network from {path}')
    self.history_network.load_state_dict(torch.load(path))
    self.history_network.to(device)
    self.history_network.eval()
    self.use_history_network = True
    dummy_input = np.random.choice([-1, 0, 1], size=(1,5,8,8))
    value, policy = self.history_network(torch.tensor(dummy_input, dtype=torch.float32).to(device))

  def latest_network(self) -> AlphaZeroNet:
    if self._networks:
      return copy.deepcopy(self._networks[max(self._networks.keys())].to(device))
    elif self.use_history_network:
      return self.history_network.to(device)
    else:
      return make_uniform_network().to(device)

  def save_network(self, step: int, network: AlphaZeroNet):
    self._networks[step] = copy.deepcopy(network) # Caution: this is overwriting the network at step if it already exists
    self._networks[step].eval()
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
    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/value_loss.npy', np.array(self.value_loss))
    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/policy_loss.npy', np.array(self.policy_loss))
    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/total_loss.npy', np.array(self.total_loss))

    self.play_round_policy_loss.append(np.mean(np.array(self.this_round_policy_loss)))
    self.play_round_value_loss.append(np.mean(np.array(self.this_round_value_loss)))
    self.play_round_total_loss.append(np.mean(np.array(self.this_round_total_loss)))


    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/play_round_value_loss.npy', np.array(self.play_round_value_loss))
    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/play_round_policy_loss.npy', np.array(self.play_round_policy_loss))
    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/play_round_total_loss.npy', np.array(self.play_round_total_loss))

    np.save(f'{self.save_folder}/self_play_round_{self.self_play_round}/self_play_times.npy', np.array(self.self_play_times))

    self.this_round_value_loss = []
    self.this_round_policy_loss = []
    self.this_round_total_loss = []

  def record_loss(self, value_loss, policy_loss, total_loss):
    # index is total number of training steps = config.training_steps * config.max_iter
    self.value_loss.append(value_loss)
    self.policy_loss.append(policy_loss)
    self.total_loss.append(total_loss)

    self.this_round_value_loss.append(value_loss)
    self.this_round_policy_loss.append(policy_loss)
    self.this_round_total_loss.append(total_loss)

  def save_loss(self):
    np.save(f'{self.save_folder}/value_loss.npy', np.array(self.value_loss))
    np.save(f'{self.save_folder}/policy_loss.npy', np.array(self.policy_loss))
    np.save(f'{self.save_folder}/total_loss.npy', np.array(self.total_loss))

    np.save(f'{self.save_folder}/play_round_value_loss.npy', np.array(self.play_round_value_loss))
    np.save(f'{self.save_folder}/play_round_policy_loss.npy', np.array(self.play_round_policy_loss))
    np.save(f'{self.save_folder}/play_round_total_loss.npy', np.array(self.play_round_total_loss))