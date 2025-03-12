from config import AlphaZeroConfig
from game import GomokuGame, Node
from replay_buffer import ReplayBuffer
from sharedstorage import SharedStorage
from network import *
import math
from typing import List
import random
import numpy as np

# AlphaZero training is split into two independent parts: AlphaZeroNet training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.

# def alphazero(config: AlphaZeroConfig):
#   storage = SharedStorage()
#   replay_buffer = ReplayBuffer(config)

#   for i in range(config.num_actors):
#     # TODO: lauch_job function used to parellel computation
#     launch_job(run_selfplay, config, storage, replay_buffer)

#   train_network(config, storage, replay_buffer)

#   return storage.latest_network()


##################################
####### Part 1: Self-Play ########

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    assert isinstance(x, np.ndarray), f"np2torch expected 'np.ndarray' but received '{type(x).__name__}'"
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  for i in range(2):
    # print(f'==============begin game {i}==============')
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: AlphaZeroNet):
  game = GomokuGame()

  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)

  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: GomokuGame, network: AlphaZeroNet):
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: GomokuGame, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.items()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: GomokuGame, network: AlphaZeroNet):
  obs = np2torch(game.make_image(-1)).unsqueeze(0)
  value, policy_logits = network.inference(obs)

  # Expand the node.
  policy_logits = policy_logits.detach().cpu().numpy()[0]
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

# Sample from softmax distribution of N(s,a)
def softmax_sample(visit_counts):
    total = sum([count for count, _ in visit_counts])
    probs = [count / total for count, _ in visit_counts]
    actions = [action for _, action in visit_counts]
    action = random.choices(actions, weights=probs)[0]
    return probs, action


##################################
####### Part 2: Training #########
# TODO: Modify logic of training network
def train_network(config: AlphaZeroConfig, storage: SharedStorage,
          replay_buffer: ReplayBuffer):
  # network = AlphaZeroNet().to(device)
  network = storage.latest_network()
  network.train()
  optimizer = torch.optim.SGD(network.parameters(), 
                              lr=2e-1, 
                              momentum=config.momentum, weight_decay=config.weight_decay)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    for image, (target_value ,target_policy) in batch:
      image_tensor = np2torch(image).unsqueeze(0)
      target_value = torch.tensor(target_value, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
      target_policy = np2torch(np.array(target_policy))
      value, policy_logits = network.inference(image_tensor)
      policy_logits = policy_logits.squeeze(0)

      loss = torch.nn.functional.mse_loss(value, target_value) + \
             torch.nn.functional.cross_entropy(policy_logits, target_policy)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    # update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network.eval())



# def train_network(config: AlphaZeroConfig, storage: SharedStorage,
#                   replay_buffer: ReplayBuffer):
#   network = AlphaZeroNet()
#   optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
#                                          config.momentum)
#   for i in range(config.training_steps):
#     if i % config.checkpoint_interval == 0:
#       storage.save_network(i, network)
#     batch = replay_buffer.sample_batch()
#     update_weights(optimizer, network, batch, config.weight_decay)
#   storage.save_network(config.training_steps, network)

# def update_weights(optimizer: tf.train.Optimizer, network: AlphaZeroNet, batch,
#                    weight_decay: float):
#   loss = 0
#   for image, (target_value, target_policy) in batch:
#     value, policy_logits = network.inference(image)
#     loss += (
#         tf.losses.mean_squared_error(value, target_value) +
#         tf.nn.softmax_cross_entropy_with_logits(
#             logits=policy_logits, labels=target_policy))

#   for weights in network.get_weights():
#     loss += weight_decay * tf.nn.l2_loss(weights)

#   optimizer.minimize(loss)