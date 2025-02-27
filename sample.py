from gomoku_env import GomokuEnv
import pickle

def sample_full_trajectory(num_episodes, board_size = 8):
    env = GomokuEnv(board_size)
    replay_buffer = []

    #  Maximum number of timesteps each game
    max_steps_per_game = board_size ** 2

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        # trajectory (s, a, r, s', done)
        trajectory = []  

        for step in range(max_steps_per_game):
            # Random Actions
            action = env.action_space.sample()  
            next_state, reward, done, info = env.step(action)  

            trajectory.append((state, action, reward, next_state, done))
            
            # update state
            state = next_state

            # End the game if finished
            if done:
                break

        replay_buffer.append(trajectory)

    print(f"Finish sampling with {len(replay_buffer)} trajectories")

    with open("gomoku_trajectories.pkl", "wb") as f:
        pickle.dump(replay_buffer, f)


if __name__ == "__main__":
    sample_full_trajectory(100000)
    
    # Read Data
    # with open("gomoku_trajectories.pkl", "rb") as f:
    #     loaded_replay_buffer = pickle.load(f)