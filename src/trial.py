import chess
from chess import Move
import random
import config
import numpy as np
# original_node = Node(chess.STARTING_FEN)
# for _ in range(100):
#     print(_)
#     root_node = original_node
#     path : list[Edge] = []
#     while not root_node.is_leaf_node():
#         root_node.get_child_edges()
#         child_edges = root_node.child_edges
#         # UCB_vals = [edge.UCB() for edge in child_edges]
#         best_edge = child_edges[random.randint(0,len(child_edges))-1]
#         path.append(best_edge)
#         root_node = best_edge.output_node
#         if original_node == root_node:
#             print("YES")
#     for edge in path:
#         edge.output_node.N += 1
#         edge.N += 1

# NUM = original_node.child_edges
# NUM = [J.N for J in NUM]
# print(*NUM)

# points = {"a" : 1, "b" : 2, "c" : 3, "d" : 4, "e" : 5, "f" : 6, "g" : 7, "h" : 8}
# str = "e2e4"
# str = [points[_]-1 if _ in points.keys() else int(_) for _ in str]
# print(str)

# print((str(chess.Board().piece_at(4))))

# board = chess.Board()
# board.push(chess.Move.from_uci("e2e4"))
# board.push(chess.Move.from_uci("e7e5"))
# en_passant_square = board.ep_square
# print(en_passant_square)
# print(type(en_passant_square))

# print(np.random.choice([0,1,109,78,23]))

# _dict = {
#     "b" : 2,
#     "a" : 1,
#     "c" : 3,
#     "d" : 4
# }

# print(_dict.keys())
# print(_dict.values())
# for _ in range(1, 4):
#     win = np.load(f"/home/atharva128/RLsoc/chess_engine/selfPlayData/game_{_}/win.npy")
# board = np.load("/home/atharva128/RLsoc/chess_engine/saved_games/game_2/board.npy")
# prob_label = np.load("/home/atharva128/RLsoc/chess_engine/saved_games/game_2/prob_label.npy")
    # print(win)
# print(board.shape)
# print(prob_label.shape)

# np.save("delete.npy", np.array([1,2,4,5]))
# np.save("delete.npy", 2)
# np.save("delete.npy", np.ones((3,4)))

# r1 = np.load("delete.npy")
# r2 = np.load("delete.npy")
# r3 = np.load("delete.npy")

# print(r1)
# print(r2)
# print(r3)

# import os
# os.makedirs("/home/atharva128/RLsoc/chess_engine/selfPlayData/1")

# import sys
# sys.stdout.write("\r")
# for _ in range(1,10001):
#     sys.stdout.write("\r"+str(_))
    # sys.stdout.flush()
# num = int(input())

# sum_ = 0
# for num in range(1,20):
#     board = np.load(f"./saved_games/game_{num}/board.npy")
#     # prob = np.load(f"./saved_games/game_{num}/prob_label.npy")

#     sum_ += board.shape[0]
# print(sum_)

# for num in range(1, 20):
#     win = np.load(f"./saved_games/game_{num}/win.npy")
#     win = int(win)
#     print(win)

# board = np.load("./continuous_training_saved_games/game_1/board.npy")
# print(board.shape)
# prob = np.load("./continuous_training_saved_games/game_1/prob_label.npy")
# print(prob.shape)
# print(np.where(board[0][11]!=0))
# print(np.where(prob[0]!=0))
# print(prob)

