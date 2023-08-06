import chess
from node_edge import Node, Edge
import config
import numpy as np
import sys
from ChessEnv import neural_input, get_actions

class MCTS:
    
    def __init__(self, NN, starting_fen = chess.STARTING_FEN):
        sys.stdout.write(f"\rInitiating MCTS algorithm for {config.MCTS_EPOCHS} iterations.")
        self.trajectory = []
        self.root_node = Node(starting_fen)
        self.NN = NN
        self.trajectory : list[Edge] = []
        self.points = {"a" : 1, "b" : 2, "c" : 3, "d" : 4, "e" : 5, "f" : 6, "g" : 7, "h" : 8}
        self.train()
        sys.stdout.write(f"\rMCTS algorithm succesfully ran for {config.MCTS_EPOCHS} times.")

    def selection(self, curr_node : Node) -> Node:
        truck = 1
        sys.stdout.write("\rPerforming selection step.")
        while not curr_node.is_leaf_node():
            child_edges = curr_node.child_edges
            UCB_vals = [edge.UCB() for edge in child_edges]
            key = [index for index, element in enumerate(UCB_vals) if element == max(UCB_vals)]
            best_edge : Edge = child_edges[np.random.choice(key)]
            self.trajectory.append(best_edge)
            truck += 1
            curr_node = best_edge.output_node
        return curr_node

    def expansion(self, curr_node : Node) -> Node:
        sys.stdout.write("\rExpanding the MCTS tree (Expansion step).")
        board = curr_node.board
        possible_actions = list(board.generate_legal_moves())
        if len(possible_actions) == 0:
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                curr_node.value = 0
            else:
                curr_node.value = 1 if outcome.winner == chess.WHITE else -1
            return curr_node
        
        input_NN = neural_input(curr_node.state)
        y_pred = self.NN.forward(input_NN)
        p, v = np.array(y_pred[0].detach()).reshape(73, 8, 8), np.array(y_pred[1].detach())[0][0]
        actions = get_actions(p, curr_node.state)
        curr_node.value = v
        curr_node.N += 1

        for action in possible_actions:
            curr_node.add_child(action, actions[action.uci()])
        return curr_node

    def backpropogation(self, value : float):
        sys.stdout.write("\rPerforming backpropogation step.")
        for edge in self.trajectory:
            edge.output_node.N += 1
            edge.N += 1
            edge.value += value
        
        ls = [edge.N for edge in self.trajectory]

    def train(self):
        for _ in range(config.MCTS_EPOCHS):
            sys.stdout.write(f"\rDoing MCTS, epoch : {_+1}/{config.MCTS_EPOCHS}")
            self.trajectory = []
            curr_node = self.root_node
            curr_node = self.selection(self.root_node)
            self.root_node.N += 1
            curr_node = self.expansion(curr_node)
            self.backpropogation(curr_node.value)
        the_ls = [edge.N for edge in self.root_node.child_edges]