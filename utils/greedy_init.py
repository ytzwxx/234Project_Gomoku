import numpy as np
import math

def get_best_move_greedy(env):
    """
    Use simple greedy policy to find best next move.
    If it is possible to win in next move, make a move and win.
    Otherwise evaluate the board and choose the move with highest score.
    
    Parameters:
        env (GomokuEnv): Gomoku Environment
    Returns:
        possible_moves tuple[int, int]: tuple of integers representing the move position under this greedy policy.
    """
    board = env.board.copy()
    board_size = env.board_size
    possible_moves = get_possible_moves(board, board_size)
    
    # check whether it is possible to win in next move
    for row, col in possible_moves:
        checking_board = board.copy()
        checking_board[row, col] = 1
        if check_win(checking_board, 1, board_size):
            return row * board_size + col

    #  Otherwise evaluate the board and choose the move with highest score.
    best_move = None
    best_score = -math.inf
    for row, col in possible_moves:
        checking_board = board.copy()
        checking_board[row, col] = 1
        score = evaluate(checking_board)
        if score > best_score:
            best_score = score
            best_move = row * board_size + col
    return best_move

def get_possible_moves(board, board_size):
    """
    To simplify computation, only check moves that near existing placements.

    Parameters:
        board (np.ndarray): 2-d array retreive from GomokuEnv
        board_size (int): board size retreive from GomokuEnv
    Return:
        possible_moves (Set): set of all possible moves. Moves are in format pf (x: int, y: int).
    """
    isempty = True
    possible_moves = set()
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] != 0:
                isempty = False
                for v_move in [-1, 1]:
                    for h_move in [-1, 1]:
                        v, h = i + v_move, j + h_move
                        if 0 <= v < board_size and 0 <= h < board_size and board[v, h] == 0:
                            possible_moves.add((v, h))
    # If the board is empty, return the middle of board
    if isempty == True:
        center = board_size // 2
        possible_moves.add((center, center))
    return possible_moves

def check_win(board, player, board_size):
    """
    Check whether current player can win in this move

    Parameters:
        board (np.ndarray): 2-d array retreive from GomokuEnv
        board_size (int): board size retreive from GomokuEnv
    Return:
        bool: true indicates current player can win in this move; false otherwise.
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == player:
                for dr, dc in directions:
                    count = 1
                    r, c = i + dr, j + dc
                    while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                        count += 1
                        r += dr
                        c += dc
                    r, c = i - dr, j - dc
                    while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                        count += 1
                        r -= dr
                        c -= dc
                    if count >= 5:
                        return True
    return False



def evaluate(board):
    """
    Evaluate every lines on board in all directions.
    If some specific "winning" patterns exist, evaluate for higher score.

    Parameters:
        board (np.ndarray): 2-d np array of the board
    Returns:
        score (int): score for this move
    """
    score = 0
    board_size = board.shape[0]

    # check patterns vertially & horizontally
    for i in range(board_size):
        score += check_pattern(board[i, :])
        score += check_pattern(board[:, i])
    # check patterns in two diagonal directions
    for offset in range(-board_size + 1, board_size):
        main_diagonal = board.diagonal(offset)
        score += check_pattern(main_diagonal)

        anti_diagonal = np.fliplr(board).diagonal(offset)
        score += check_pattern(anti_diagonal)
    return score

def check_pattern(line):
    '''
    Help function to check some possible  "winning" patters in evaluate function.

    Parameters:
        line (np.ndarray): 1-d array representing each line in board
    Returns:
        score (int): score for this line
    '''

    line_str = "".join(str(x) for x in line)
    # Simple five
    if "11111" in line_str:
        return 200
    # Double four
    elif "011110" in line_str or :
        return 50
    # Simple four
    elif "11110" in line_str or "01111" in line_str or "11011" in line_str:
        return 60
    # Double three
    elif "01110" in line_str:
        return 25
    elif "11100" in line_str or "00111" in line_str or "10110" in line_str:
        return 10
    elif "11000" in line_str or "01100" in line_str or "00110" in line_str or  "00011" in line_str:
        return 5
    return 0




