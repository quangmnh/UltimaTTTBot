# 
#        All bugs locked in hell already.
#                Don't worry.
#                       /
#                      /
#             )            (
#            /(   (\___/)  )\
#           ( #)  \ ('')| ( #
#            ||___c\  > '__||
#            ||**** ),_/ **'|
#      .__   |'* ___| |___*'|
#       \_\  |' (    ~   ,)'|
#        ((  |' /(.  '  .)\ |
#         \\_|_/ <_ _____> \______________
#          /   '-, \   / ,-'      ______  \
#         /      (//   \\)     __/     /   \
#                             './_____/


import numpy as np
from numpy.core.fromnumeric import cumprod
from numpy.core.shape_base import block
from state import State, UltimateTTT_Move
import random
inff = 10000000
class agent:
    def __init__(self) -> None:
        self.isFirst = False
        self.center_placed = 0
        self.o_trap = {0,1,2,3,5,6,7,8}
        self.isEndGame = False
        self.prev_block = 0
        self.next_block = 8

        self.score = 32
        self.inf = 1000000
        self.inff = 10000000
        self.global_scale = 23
        self.winning_pattern = {
            (0,0): (((0,1),(0,2)),((1,0),(2,0)),((1,1),(2,2))),
            (1,0): (((0,0),(2,0)),((1,1),(1,2))),
            (2,0): (((0,0),(1,0)),((0,2),(1,1)),((2,1),(2,2))),
            (0,1): (((0,0),(0,2)),((1,1),(2,1))),
            (1,1): (((0,0),(2,2)),((2,0),(0,2)),((0,1),(2,1)),((1,0),(1,2))),
            (2,1): (((0,1),(1,1)),((2,0),(2,2))),
            (0,2): (((0,0),(0,1)),((1,2),(2,2)),((1,1),(2,0))),
            (1,2): (((1,0),(1,1)),((0,2),(2,2))),
            (2,2): (((2,0),(2,1)),((0,2),(1,2)),((0,0),(1,1)))
        }
        self.X = State.X
        self.O = State.O
        self.cell_scale_add = {
            (0,0): 3,
            (1,0): 2,
            (2,0): 3,
            (0,1): 2,
            (1,1): 4,
            (2,1): 2,
            (0,2): 3,
            (1,2): 2,
            (2,2): 3
        }

        self.cells = self.cell_scale_add.keys()

        self.block_scale_mult ={
            0:1.3,
            1:1.2,
            2:1.3,
            3:1.2,
            4:1.4,
            5:1.2,
            6:1.3,
            7:1.2,
            8:1.3
        }

        self.block_scale_add ={
            0:3,
            1:2,
            2:3,
            3:2,
            4:4,
            5:2,
            6:3,
            7:2,
            8:3
        }
        
        self.total = sum(self.block_scale_add.values())

        self.winning_block ={
            0:((1,2), (3,6), (4,8)),
            1:((0,2), (4,7)),
            2:((0,1), (6,4), (5,8)),
            3:((0,6), (4,5)),
            4:((0,8), (2,6), (1,7), (3,5)),
            5:((2,8), (3,4)),
            6:((0,3), (7,8)),
            7:((1,4), (6,8)),
            8:((0,4), (6,7), (2,5)),
        }

        self.winning_block_sequence = np.array(((0,1,2), (0,3,6), (0,4,8), (1,4,7), (2,4,6), (2,5,8), (3,4,5), (6,7,8)))

    def reset(self):
        self.isFirst = False
        self.center_placed = 0
        self.o_trap = {0,1,2,3,5,6,7,8}
        self.isEndGame = False
        self.prev_block = 0
        self.next_block = 8

    def board_eval(self, board: np.ndarray, player):
        score = 0
        for sequence in self.winning_block_sequence:
            line = board[sequence]
            if (player==line).any():
                if (-player==line).any():
                    continue
                if np.sum(player==line)>1:
                    score += self.total/3-1
                score += 1
            elif (-player==line).any():
                if np.sum(-player==line)>1:
                    score -= self.total/3-1
                score -= 1
        return score

    def heuristic(self, cur_state: State, player):
        if cur_state.game_result(cur_state.global_cells.reshape(3,3)) == -player:
            return self.inf + len(np.where(cur_state.blocks==0)[0])
        elif cur_state.game_result(cur_state.global_cells.reshape(3,3)) == player:
            return -self.inf - len(np.where(cur_state.blocks==0)[0])
        
        if len(np.where(cur_state.blocks==0)[0])==0:
            return 0

        score = self.board_eval(cur_state.global_cells, player) * self.global_scale

        for block in cur_state.blocks:
            if len(np.where(block==0)[0])!=0:
                score += self.board_eval(block.ravel(), player)
        return score
    def opponent(self, cur_state:State, depth, alpha= -inff, beta = inff):
        if depth == 0 or cur_state.game_over or len(np.where(cur_state.blocks==0)[0])==0:
            return self.heuristic(cur_state, cur_state.player_to_move)
        score = self.inff
        for valid_move in cur_state.get_valid_moves:
            temp_state = State(cur_state)
            temp_state.free_move = cur_state.free_move
            temp_state.act_move(valid_move)
            score = min(score, self.player(temp_state, depth-1, alpha, beta))
            if score <= alpha:
                return score
            beta = min(beta, score)
        return score

    def player(self, cur_state:State, depth, alpha= -inff, beta = inff):
        if depth == 0 or cur_state.game_over or len(np.where(cur_state.blocks==0)[0])==0:
            return self.heuristic(cur_state, cur_state.player_to_move)
        score = -self.inff
        for valid_move in cur_state.get_valid_moves:
            temp_state = State(cur_state)
            temp_state.free_move = cur_state.free_move
            temp_state.act_move(valid_move)
            score = max(score, self.opponent(temp_state, depth, alpha, beta))
            if score >=beta:
                return score
            alpha = max(alpha, score)
        return score

    def run_minimax(self,cur_state: State, depth):
        best_moves = []
        for valid_move in cur_state.get_valid_moves:
            temp_state = State(cur_state)
            temp_state.free_move = cur_state.free_move
            temp_state.act_move(valid_move)
            score = self.opponent(temp_state, depth)
            best_moves.append((valid_move, score))
        move, best_score = max(best_moves, key=lambda x: x[1])
        return random.choice([best_move for best_move, val in best_moves if val==best_score])

def select_move(cur_state:State, remain_time):
    moves = cur_state.get_valid_moves
    if len(moves) == 81:
        a.reset()
        a.isFirst = True
        a.center_placed+=1
        return moves[40]
    if len(np.where(cur_state.blocks==0)[0])==80 and not a.isFirst:
        a.reset()
    if a.isFirst:
        if a.center_placed<=7:
            a.center_placed+=1
            return moves[4]
        elif a.center_placed==8:
            a.center_placed+=1
            for move in moves:
                if move.x*3+move.y == move.index_local_board:
                    a.prev_block = move.index_local_board
                    for seq in a.winning_block_sequence:
                        if a.prev_block in seq and 4 in seq:
                            for i in seq:
                                if i!=4 and i!= a.prev_block:
                                    a.next_block = i
                    a.o_trap.remove(a.prev_block)
                    a.o_trap.remove(a.next_block)
                    return move
        else:
            if cur_state.game_result(cur_state.blocks[a.next_block])==1:
                # print('End game')
                for move in moves:
                    if move.x*3+move.y==a.prev_block or move.x*3+move.y==a.next_block:
                        return move
            else:
                # print('middle game ', moves[0].index_local_board)
                if cur_state.previous_move.x*3+cur_state.previous_move.y in a.o_trap:
                    for move in moves:
                        if move.x*3+move.y == a.prev_block and move.index_local_board in a.o_trap:
                            return move
                else:
                    for move in moves:
                        if move.index_local_board == a.next_block and move.x*3+move.y==a.prev_block:
                            temp_state = State(cur_state)
                            temp_state.free_move = cur_state.free_move
                            temp_state.act_move(move)
                            return move
                    for move in moves:
                        if move.index_local_board == a.next_block and move.x*3+move.y==a.next_block:
                            temp_state = State(cur_state)
                            temp_state.free_move = cur_state.free_move
                            temp_state.act_move(move)
                            return move
                
    else:
        temp_state = State(cur_state)
        temp_state.free_move = cur_state.free_move
        bestMove = a.run_minimax(temp_state, 2)
        # print(bestVal)
        if bestMove:
            return bestMove
    return None

a = agent()