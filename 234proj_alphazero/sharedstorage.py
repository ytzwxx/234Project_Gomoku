from network import Network

# TODO
def make_uniform_network(self):
    pass

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.iterkeys())]
    else:
      return make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network