from game import Game
from model import CustomModel
import chess
import chess.svg
from chess import Move
import torch
import os
model = CustomModel()
model.load_state_dict(torch.load("./model.pth"))

game = Game(model=model)
id = 1 
while not game.terminated()[0]:
    player_move = input("Enter the move : ")
    game.make_move(Move.from_uci(player_move))
    board_svg = chess.svg.board(board=game.board)
    with open(f"./gameImages/{id}.svg", "w") as svg_file:
        svg_file.write(board_svg)
    id += 1