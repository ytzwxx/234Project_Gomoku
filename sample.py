from gomoku_env import GomokuEnv
import pickle
import numpy as np
from utils.greedy_init import get_best_move_greedy

def sample_full_trajectory(num_episodes, board_size = 8):
    env = GomokuEnv(board_size)
    replay_buffer = []

    #  Maximum number of timesteps each game
    max_steps_per_game = board_size ** 2

    for episode in range(num_episodes):
        if episode % 10000 == 0:
            print(f'finish {episode} episodes' )
        state = env.reset()
        done = False

        # trajectory (s, a, r, s', done)
        trajectory = np.zeros((max_steps_per_game, 5), dtype=object) 

        for step in range(max_steps_per_game):
            # TODO: Change stratagy if needed.
            # Use greedy policy here.
            action = get_best_move_greedy(env=env)
            next_state, reward, done, info = env.step(action)  

            trajectory[step] = (state, action, reward, next_state, done)
            
            # update state
            state = next_state

            # End the game if finished
            if done:
                # Delete timesteps that are not used
                trajectory = trajectory[:step+1]
                break
 
        replay_buffer.append(trajectory)

    print(f"Finish sampling with {len(replay_buffer)} trajectories")

    print("Saving data...")
    replay_buffer = np.array(replay_buffer, dtype=object)
    np.savez_compressed("gomoku_trajectories.npz", replay_buffer=replay_buffer)


if __name__ == "__main__":
    sample_full_trajectory(10)