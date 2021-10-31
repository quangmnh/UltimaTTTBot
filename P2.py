# //                   _ooOoo_
# //                  o8888888o
# //                  88" . "88
# //                  (| -_- |)
# //                  O\  =  /O
# //               ____/`---'\____
# //             .'  \\|     |//  `.
# //            /  \\|||  :  |||//  \
# //           /  _||||| -:- |||||-  \
# //           |   | \\\  -  /// |   |
# //           | \_|  ''\---/''  |   |
# //           \  .-\__  `-`  ___/-. /
# //         ___`. .'  /--.--\  `. . __
# //      ."" '<  `.___\_<|>_/___.'  >'"".
# //     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
# //     \  \ `-.   \_ __\ /__ _/   .-` /  /
# //======`-.____`-.___\_____/___.-`____.-'======
# //                   `=---='
# //
# //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# //          佛祖保佑           永无BUG
# //         God Bless        Never Crash

import numpy as np
from state import State, State_2, UltimateTTT_Move
import sys
import random

APPROXIMATE_WIN_SCORE = 7
BIG_BOARD_WEIGHT = 23
WIN_SCORE = 10**6
POSSIBLE_WIN_SEQUENCES = np.array ([
                            [0, 1, 2], 
                            [3, 4, 5], 
                            [6, 7, 8], 
                            [0, 3, 6], 
                            [1, 4, 7], 
                            [2, 5, 8], 
                            [0, 4, 8], 
                            [2, 4, 6]
                        ])
ALPHA_BETA_DEPTH = 3
SCORE_PER_CELL = {
                    0: 3, 
                    1: 2, 
                    2: 3, 
                    3: 2, 
                    4: 5, 
                    5: 2, 
                    6: 3, 
                    7: 2, 
                    8: 3
                }

def _legal_cells(block):
    block_flatten = block.ravel()
    return len([i for i in range(9) if block_flatten[i] == 0.0])

def _is_block_full(block):
    return _legal_cells(block) == 0

def _is_gboard_full(blocks):
    _available_blocks = []
    for i, block in enumerate(blocks):
        if not _is_block_full(block):
            _available_blocks.append(i)
    return len(_available_blocks) == 0

def _is_terminal_state(cur_state):
    return cur_state.game_over or _is_gboard_full(cur_state.blocks)

def _terminal_test(cur_state, depth):
    return _is_terminal_state(cur_state) or depth == 0

def _assess_global(block_flatten, player):
    player_counter = 0
    opponent_counter = 0
    opponent = player * -1
    for seq in POSSIBLE_WIN_SEQUENCES:
        filtered_seq = []
        filtered_indices = []
        for index in seq:
            if block_flatten[index] != 0.0:
                filtered_seq.append(block_flatten[index])
                filtered_indices.append(index)
        if len(filtered_seq) == 0:
            continue
        if player in filtered_seq:
            if opponent in filtered_seq:
                continue
            if len(filtered_seq) > 1:
                player_counter += APPROXIMATE_WIN_SCORE
            if len(filtered_seq) == 3:
                player_counter *= APPROXIMATE_WIN_SCORE                 
            player_counter += 1
        elif opponent in filtered_seq:
            if len(filtered_seq) > 1:
                opponent_counter += APPROXIMATE_WIN_SCORE
            if len(filtered_seq) == 3:
                opponent_counter *= APPROXIMATE_WIN_SCORE
            opponent_counter += 1
    return player_counter - opponent_counter

def _assess_block(block_flatten, player):
    player_counter = 0
    opponent_counter = 0
    opponent = player * -1
    for seq in POSSIBLE_WIN_SEQUENCES:
        filtered_seq = []
        filtered_indices = []
        for index in seq:
            if block_flatten[index] != 0.0:
                filtered_seq.append(block_flatten[index])
                filtered_indices.append(index)
        if player in filtered_seq:
            if opponent in filtered_seq:
                continue
            if len(filtered_seq) > 1:
                player_counter += APPROXIMATE_WIN_SCORE
            if len(filtered_seq) == 3:
                player_counter += APPROXIMATE_WIN_SCORE
            player_counter += 1
        elif opponent in filtered_seq:
            if len(filtered_seq) > 1:
                opponent_counter += APPROXIMATE_WIN_SCORE
            if len(filtered_seq) == 3:
                opponent_counter += APPROXIMATE_WIN_SCORE
            opponent_counter += 1
    return player_counter - opponent_counter


def _eval_state(cur_state, player):
    if cur_state.game_result(cur_state.global_cells.reshape(3,3)) != None:
        winner = cur_state.game_result(cur_state.global_cells.reshape(3,3))
        free_cells = 0 
        for block in cur_state.blocks:
            free_cells += _legal_cells(block)
        return (WIN_SCORE + free_cells) if (winner == -player) else (-WIN_SCORE - free_cells)
    if _is_gboard_full(cur_state.blocks):
        return 0
    ret = _assess_global(cur_state.global_cells, player) * BIG_BOARD_WEIGHT
    for i,block in enumerate(cur_state.blocks):
        if not _is_block_full(block) and cur_state.game_result(block) == None:
        # if not _is_block_full(block):
            ret += _assess_block(block.ravel(), player)
    return ret


def _generate_succ(cur_state, move):
    new_state = State(cur_state)
    new_state.free_move = cur_state.free_move
    new_state.act_move(move)
    return new_state


def _min_val_ab(cur_state, depth, alpha=-sys.maxsize-1, beta=sys.maxsize):
    if _terminal_test(cur_state, depth):
        return _eval_state(cur_state, cur_state.player_to_move)
    val = sys.maxsize
    for move in cur_state.get_valid_moves:
        successor_state = _generate_succ(cur_state, move)
        val = min(val, _max_val_ab(successor_state, depth-1, alpha, beta))
        if val <= alpha:
            return val
        beta = min(beta, val)
    return val


def _max_val_ab(cur_state, depth, alpha=-sys.maxsize-1, beta=sys.maxsize):
    if _terminal_test(cur_state, depth):
        return _eval_state(cur_state, cur_state.player_to_move)
    val = -sys.maxsize-1
    for move in cur_state.get_valid_moves:
        successor_state = _generate_succ(cur_state, move)
        val = max(val, _min_val_ab(successor_state, depth, alpha, beta))
        if val >= beta:
            return val
        alpha = max(alpha, val)
    return val

def _run_AB(cur_state, DEPTH):
    moves_res = []
    for move in cur_state.get_valid_moves:
        successor_state = _generate_succ(cur_state, move)
        weight = _min_val_ab(successor_state, DEPTH)
        moves_res.append((move, weight))

    move, best_val = max(moves_res, key=lambda x: x[1])
    return random.choice([best_move for best_move, val in moves_res if val==best_val])

# GOOD
def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves
    jump = random.random()
    if len(valid_moves) != 0:
        if cur_state.previous_move == None:
            return _run_AB(cur_state,1)
        # return _run_AB(cur_state,1)
        return _run_AB(cur_state,2)
    return None

# def select_move(cur_state, remain_time):
#     valid_moves = cur_state.get_valid_moves
#     jump = random.random()
#     if len(valid_moves) != 0:
#         if cur_state.previous_move == None:
#             return _run_AB(cur_state,1)
#         return _run_AB(cur_state,1)
#         # return _run_AB(cur_state,2)
#     return None