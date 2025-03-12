from network import *

# TODO
def make_uniform_network():
    uniform_network = AlphaZeroNet()
    uniform_network.uniform_initialization()
    return uniform_network

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> AlphaZeroNet:
    if self._networks:
      return self._networks[max(self._networks.iterkeys())].to(device)
    else:
      return make_uniform_network().to(device)

  def save_network(self, step: int, network: AlphaZeroNet):
    self._networks[step] = network
  
  def get_num_networks(self):
    return len(self._networks)