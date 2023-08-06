import chess
from chess import Move
from ChessEnv import neural_input, get_actions
import numpy as np

class Game:
    def __init__(self, model, turn : bool = 1) -> None:
        self.turn = turn
        self._id = 1
        self.model = model
        self.init_game()

    def init_game(self):
        self.board = chess.Board()
    
    def make_move(self, move : Move):
        self.board.push(move)
        if not self.terminated()[0]:
            self.make_game_move()
    
    def make_game_move(self):
        y = self.model.forward(neural_input(self.board.fen()))
        policy, value = np.array(y[0].detach()).reshape(73, 8, 8), np.array(y[1].detach())[0][0]
        actions = get_actions(policy, self.board.fen())
        keys = list(actions.keys())
        values = list(actions.values())
        key = [index for index, element in enumerate(values) if element == (min(values) if self.turn else max(values))]
        move = keys[np.random.choice(key)]
        move = Move.from_uci(move)
        self.board.push(move)
    
    def terminated(self) -> tuple():
        if not self.board.is_game_over(claim_draw=True):
            return (False, 0)
        elif not self.board.is_game_over():
            return (True, 0)
        else:
            turn = (self.board.turn == chess.WHITE)
            return (True, 1 if (turn^self.turn) else -1)
    
    def show(self):    
        board_svg = chess.svg.board(self.board)
        with open(f"/home/atharva128/RLsoc/chess_engine/gameImages/{self._id}.svg", 'w') as f:
            f.write(board_svg)
        self._id += 1