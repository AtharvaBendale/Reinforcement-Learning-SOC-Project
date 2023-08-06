import chess
from chess import Move
import config
import math


class Node:
    def __init__(self, fen : str) -> None:
        self.state = fen
        self.board = chess.Board(fen)
        self.turn = self.board.turn == 'w'  # True if white's turn
        self.child_edges : list["Edge"] = []
        self.N = 0  # Visit count
        self.value = 0 # Value for this node
        self.check = 0
        self.check2 = 0

    def __eq__(self, other : "Node"):
        return self.state == other.state    

    def make_move(self, action : Move):
        output_board = self.board.copy()
        output_board.push(action)
        output_node = Node(output_board.fen())
        return output_node

    def is_leaf_node(self):
        return self.board.is_game_over(claim_draw=True) or (self.N == 0)
    
    def add_child(self, action: Move, prob: float):
        present_moves = [edge.action for edge in self.child_edges]
        if action not in present_moves:
            edge = Edge(self, action, prob)
            self.child_edges.append(edge)


class Edge:

    def __init__(self, input_node : "Node", action : Move, prob : float = 0):
        
        self.input_node = input_node
        self.action = action
        output_board = self.input_node.board.copy()
        output_board.push(action)
        self.output_node = Node(output_board.fen())
        self.prob = prob
        self.Nparent = self.input_node.N
        self.N = 0  # Visit count for this edge
        self.value = 0
        self.turn : bool = self.input_node.state.split(" ")[1] == "w"

    def isequal(self, edge : "Edge") -> bool:
        return self.input_node == edge.input_node and self.action == edge.action
    
    
    def UCB(self):
       self.Nparent = self.input_node.N
       return (math.log((1 + self.Nparent + config.C_base)/(config.C_base)) + config.C_init) * self.prob * math.sqrt(self.Nparent) / (1 + self.N) + (self.value/(self.N+1) if self.turn else -self.value/(self.N+1))

    def update_prob(self, new_prob):
        assert new_prob <= 1
        self.prob = new_prob
    
    def update_value(self, new_value):
        self.value = new_value