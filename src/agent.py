from mcts import MCTS
from chess import Move
from game import Game
import numpy as np
from node_edge import Node
import os, sys
import config
from ChessEnv import neural_input, get_actions, action_to_probs

class Agent:
    
    def __init__(self, CNN, epsilon = config.EPSILON, id = 1) -> None:
        sys.stdout.write(f"\rInitiating the Agent with epsilon = {epsilon}, to create dataset game_{id}.npy .")
        self.epsilon = epsilon
        self.CNN = CNN
        self._id : int = id
        self.prepare_label_files(*self.complete_game())

    def complete_game(self, turn : bool = 1) -> tuple():
        game = Game(self.CNN ,turn)
        board_fen_list : list[str] = []
        prob_labels_list : list[np.ndarray] = []
        sys.stdout.write("\rStarting Agent selfplay.")
        while not game.terminated()[0]:
            root_node : Node = MCTS(self.CNN, game.board.fen()).root_node
            prob_labels = action_to_probs(root_node=root_node)
            assert not np.all(prob_labels==0)
            
            del root_node
            board_fen_list.append(game.board.fen())
            prob_labels_list.append(prob_labels)
            y = self.CNN.forward(neural_input(game.board.fen()))
            policy, value = np.array(y[0].detach()).reshape(73, 8, 8), np.array(y[1].detach())[0][0]
            actions = get_actions(policy, game.board.fen())
            if np.random.random() < self.epsilon:
                move = np.random.choice(list(actions.keys()))
            else:
                keys = list(actions.keys())
                values = list(actions.values())
                move = keys[values.index(max(values))]
            move = Move.from_uci(move)
            game.make_move(move)
        sys.stdout.write("\rAgent selfplay done.")
        t : bool = game.terminated()[1]
        del game
        return t, board_fen_list, prob_labels_list
    
    def prepare_label_files(self, win, board_fen_list, prob_labels_list):
        sys.stdout.write(f"\rStoring the selfplay game data in game_{self._id}.npy .")
        os.makedirs(f"./saved_games/game_{self._id}")
        np.save(f"./saved_games/game_{self._id}/win.npy", win)
        np.save(f"./saved_games/game_{self._id}/board.npy", np.asarray([*[neural_input(board_fen) for board_fen in board_fen_list]]))
        np.save(f"./saved_games/game_{self._id}/prob_label.npy", np.array(prob_labels_list))
        sys.stdout.write(f"\rSelfplay game data stored in game_{self._id}.npy .")