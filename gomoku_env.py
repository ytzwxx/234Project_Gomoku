import gym
from gym import spaces
import numpy as np
import random

class GomokuEnv(gym.Env):
    """
    Gomoku env based on OpenAI Gym environment.
    Two players: player 1 and player -1
    Board stored on 2-d Numpy Array. 0 represents empty space.
    """
    # Board Visualization Mode
    metadata = {'render.modes': ['human']}
    
    # Default board size is 8*8 for simplicity
    def __init__(self, board_size=8):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        # Action space: 1-D index on board
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        # Obervation space: 3-chanel matrix with size of board_size * board_size.
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),  # board: -1, 0, 1 representing opponent player, empty, current player
            spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),    # valid_moves: 0 represents valid moves and 1 represents invalid moves
            spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)    # curretn player: all 1s if its agent round, all 0s otherwise
        ))
        self.reset()

    def get_state(self):
        """
        Create shallow copy for each chanel in current state.
        """
        board = np.copy(self.board)
        valid_moves = (self.board == 0).astype(np.int8)
        if self.current_player == 1:
            current_chanel = np.ones((self.board_size, self.board_size), dtype=np.int8)
        else:
            np.zeros((self.board_size, self.board_size), dtype=np.int8)
        return [board, valid_moves, current_chanel]
        
    def reset(self):
        """
        Reset the board
        Return deep copy of the new board.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)\
        # Assume agent moves first.
        self.current_player = 1
        self.done = False
        return self.get_state()
    
    def step(self, action):
        """
        Execute the next action.
        Parameter:
            action: int, represents 1-D index on board
        Return:
            observation: spaces.Box, Current state on the board
            reward: float, Immediate Reward
            done: Boolean, Whether the current episode of game finished
            info: {}, optional
        """
        info = {}
        # If game finish, return the current board
        if self.done:
            return np.copy(self.board), 0, True, info
        
        # Convert 1-D board to 2-D
        row = action // self.board_size
        col = action % self.board_size
        
        # check whether action is feasible
        # Apply penalty and terminate the game otherwise
        # Reward can be modified later
        if self.board[row, col] != 0:
            self.done = True
            return np.copy(self.board), -10, True, {"error": "Invalid move"}
        
        # Agent makes the next move
        self.board[row, col] = 1
        # Check whether agent wins
        # Reward can be modified later
        if self.check_win(1, row, col):
            self.done = True
            return np.copy(self.board), 1, True, {"result": "Agent wins"}
        
        # Check tie condition
        if not (0 in self.board):
            self.done = True
            return np.copy(self.board), 0, True, {"result": "Draw"}
        
        # Opponent Randome Policy, Replace later
        empty_row_indices, empty_col_indices = np.where(self.board == 0)
        empty_positions = list(zip(empty_row_indices, empty_col_indices))
        if empty_positions:
            opp_move = random.choice(empty_positions)
            self.board[opp_move] = -1
            # Check whether opponent wins
            if self.check_win(-1, opp_move[0], opp_move[1]):
                self.done = True
                return np.copy(self.board), -1, True, {"result": "Opponent wins"}
        else:
            self.done = True
            return np.copy(self.board), 0, True, {"result": "Draw"}
        
        # Game continues
        self.current_player = 1
        return self.get_state(), 0, False, info
    
    def check_win(self, player, row, col):
        """
        Given current board info and player's next move check whether the current player wins under this move
        Parameters:
            player: integer, current player (-1 or 1) or 0 represenrs empty
            row, col: (int, int), position of next move
        Return: Boolean, True if win, False otherwise
        """
        # define 4 directions
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False
    
    def render(self, mode='human'):
        """Print the current board"""
        board_str = ""
        for i in range(self.board_size):
            row_str = ""
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    row_str += "X "
                elif self.board[i, j] == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            board_str += row_str + "\n"
        print(board_str)
    
    def close(self):
        pass

if __name__ == "__main__":
    env = GomokuEnv(board_size=8)
    state = env.reset()
    print("Initial State:")
    print("Board:")
    print(state[0])
    env.render()
    
    done = False
    while not done:
        # TODO: Random Sample to check
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print("Action:", action, "Reward:", reward, "Info:", info)
        env.render()
        if done:
            print("Game Over")

