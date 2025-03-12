# TODO: Modify later.

class AlphaZeroConfig(object):

  def __init__(self):
    self.max_iter = 80
    ### Self-Play
    self.num_actors = 5

    self.parrallel_actor = 4

    self.num_sampling_moves = 3
    self.max_moves = 64
    self.num_simulations = 100 # roll out numbers

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.1  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.1

    # UCB formula
    self.pb_c_base = 3500
    self.pb_c_init = 1.25

    ### Training
    # self.training_steps = int(1e5)
    # self.checkpoint_interval = int(1e3)
    self.training_steps = int(200)
    self.checkpoint_interval = int(5) # save network every x iters
    self.window_size = int(1e6)
    self.batch_size = 128

    self.save_folder = 'training_results'
    
    self.save_play_time_interval = 5

    self.save_after_each_checkpoint_interval = True
    self.save_after_each_play_interval = True

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }