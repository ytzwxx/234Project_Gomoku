import numpy as np
import copy
import random

class Node(object):
    """
    Node of Monto Carlo Tree
    """
    def __init__(self, prior: float):
        self.visit_count = 0
        # self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        # Format of action: Node
        self.children = {}  

    # Whether the tree has been expanded
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class GomokuGame(object):
    """
    Game history.
    Action space are interges between 0 and (size * size - 1)
    """
    def __init__(self, size=8, history=None):
        self.size = size
        # History of actions. Each actions is board index (from 0 to size^2 - 1)
        self.history = history or []
        # Each element is action distribution in each timestep.
        # Size of each element is size * size
        self.child_visits = []
        # Initialized the board, 1 as current player, -1 as opponen player, 0 as empty
        self.board = np.zeros((self.size, self.size), dtype=int)

        # Rebuild the board from given history
        if len(self.history) != 0:
           self.build_state()
    

    def build_state(self):
        """
        Takle the action from the history and builf the board in the point of view of current player.
        Only call this function before taking action at current state.
        """
        current_player = self.to_play()
        is_first_player = 1

        for action in self.history:
            row = action // self.size
            col = action % self.size
            if self.board[row, col] != 0:
                raise ValueError("Illegal action: Existing stone in this position.")
            
            if current_player == is_first_player:
                self.board[row, col] = 1
            else:
                self.board[row, col] = -1
            is_first_player = -is_first_player
        

    def terminal(self):
        """
        Check whether the game end after last move.
        Call this function before move in each round.
        """
        if len(self.history) == 0:
            return False
        idx = self.history[-1]
        row = idx // self.size
        col = idx % self.size

        if self.check_win(self.board[row, col], row, col):
            return True
        if len(self.history) == self.size * self.size:
            return True
        return False

    def terminal_value(self, state_index):
        """
        Return whether the current player wins the games.
        """
        if not self.terminal():
            raise ValueError("Illegal action: Gane not END")
        if len(self.history) == self.size * self.size:
            return 0
        idx = self.history[-1]
        row = idx // self.size
        col = idx % self.size
        
        to_play = 1 if state_index % 2 == 0 else -1
        winner = self.board[row, col]
        return 1 if winner == to_play else -1
    
    def check_win(self, player, row, col):
        """
        Given current board info and player's next move, check whether the current player wins under this move
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False

    def legal_actions(self):
        """
        Return all empty(legal) positions on the board
        """
        legal = []
        for i in range(self.size * self.size):
            r = i // self.size
            c = i % self.size
            if self.board[r, c] == 0:
                legal.append(i)
        return legal

    def clone(self):
        """
        Deep Copy current game as root in MCT simulations
        """
        cloned = GomokuGame(self.size, history=list(self.history))
        return cloned

    def apply(self, action):
        '''
        Apply next action
        '''
        r = action // self.size
        c = action % self.size
        current_player = self.to_play()
        self.board[r, c] = current_player
        self.history.append(action)

    def store_search_statistics(self, root):
        """
        After simulations, calculate root's total visits and distribution of children visted
        """
        total_visits = sum(child.visit_count for child in root.children.values())
        distribution = []
        for a in range(self.size * self.size):
            if a in root.children:
                distribution.append(root.children[a].visit_count / total_visits)
            else:
                distribution.append(0)
        self.child_visits.append(distribution)


    def make_image(self, state_index: int):
        """
        Make features as input of neuro network. The shape of input is (5,size,size)
          - Channel 0: Current player's position
          - Channel 1: Opponent's position
          - Channel 2: Position of last move
          - Channel 3: Position of second last move
          - Channel 4: Whether the current player is first player
        state_index is the index(timestep) we sampled
        """
        if state_index == -1:
            board = self.board
            current_player = self.to_play()
            last = self.history[-2] if len(self.history) >= 2 else None
            second_last = self.history[-3] if len(self.history) >= 3 else None
        else:
            board = np.zeros((self.size, self.size), dtype=np.float32)
            last = self.history[state_index - 1] if len(self.history) >= 1 else None
            second_last = self.history[state_index - 2] if len(self.history) >= 2 else None
            # Build the board until state_index 
            for i, move in enumerate(self.history[:(state_index + 1)]):
                row = move // self.size
                col = move % self.size
                if i % 2 == 0:
                    player = 1
                else:
                    player = -1
                board[row, col] = player

            if state_index % 2 == 0:
                current_player = 1
            else:
                current_player = -1
        
        # Build channel 0, 1
        currentplayer_board = (board == current_player).astype(np.float32)
        opponentplayer_board = (board == -current_player).astype(np.float32)

        # Build channel 2, 3
        last_layer = np.zeros((self.size, self.size), dtype=np.float32)
        if last is not None:
            row = last // self.size
            col = last % self.size
            last_layer[row, col] = 1.0
        second_last_layer = np.zeros((self.size, self.size), dtype=np.float32)
        if second_last is not None:
            row = second_last // self.size
            col = second_last % self.size
            second_last_layer[row, col] = 1.0
        to_play_layer = np.full((self.size, self.size), current_player, dtype=np.float32)

        # Stack all five channels
        image = np.stack([currentplayer_board, opponentplayer_board, last_layer, second_last_layer, to_play_layer], axis=0)
        return image

    def make_target(self, state_index: int):
        """
        Calculate the target of Neuro Network given current game,
        Format is (Value, Action Distribution).
        """
        value = self.terminal_value(state_index)
        target_policy = self.child_visits[state_index]
        return value, target_policy

    def to_play(self):
        """
        Whether the current player is the player that move first.
        Only call this function before taking action at current state.
        """
        return 1 if (len(self.history) % 2 == 0) else -1
