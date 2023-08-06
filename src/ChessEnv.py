import chess
import numpy as np
from node_edge import Node
import sys

points = {"a" : 1, "b" : 2, "c" : 3, "d" : 4, "e" : 5, "f" : 6, "g" : 7, "h" : 8}

def f(*t) -> np.ndarray:
    output = np.zeros((8,8))
    for pos in t:
        output[pos[0]][pos[1]] = 1
    return output

def get_piece_positions(board : chess.Board) -> dict[str,list[int]]:
    piece_positions : dict[str,list[int]] = {}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece = str(piece)
            if piece in piece_positions.keys():
                piece_positions[piece].append((int(square)//8, int(square)%8))
            else:
                piece_positions[piece] = [(int(square)//8, int(square)%8)]
    for move in ['P', 'N', 'R', 'B', 'Q', 'K', 'p', 'n', 'r', 'b', 'q', 'k']:
        if move not in piece_positions.keys():
            piece_positions[move] = []

    return piece_positions

def neural_input(fen : str) -> np.ndarray:
    sys.stdout.write("\rPreparing neural input.")
    board = chess.Board(fen)
    piece_positons = get_piece_positions(board)
    t = np.asarray([np.ones((8,8)) if board.turn == chess.WHITE else np.zeros((8,8)),
                    np.ones((8,8)) if board.has_queenside_castling_rights(chess.WHITE) else np.zeros((8,8)),
                    np.ones((8,8)) if board.has_kingside_castling_rights(chess.WHITE) else np.zeros((8,8)),
                    np.ones((8,8)) if board.has_queenside_castling_rights(chess.BLACK) else np.zeros((8,8)),
                    np.ones((8,8)) if board.has_kingside_castling_rights(chess.BLACK) else np.zeros((8,8)),
                    np.ones((8,8)) if board.can_claim_fifty_moves() else np.zeros((8,8)),
                    f(*piece_positons['P']),
                    f(*piece_positons['N']),
                    f(*piece_positons['R']),
                    f(*piece_positons['B']),
                    f(*piece_positons['Q']),
                    f(*piece_positons['K']),
                    f(*piece_positons['p']),
                    f(*piece_positons['n']),
                    f(*piece_positons['r']),
                    f(*piece_positons['b']),
                    f(*piece_positons['q']),
                    f(*piece_positons['k']),
                    f((int(board.ep_square)//8, int(board.ep_square)%8)) if board.has_legal_en_passant() else np.zeros((8, 8))]
                    )
    sys.stdout.write("\rPrepared neural input.")
    del board
    return t

def get_actions(p : np.ndarray, state) -> dict[str, int]:
    sys.stdout.write("\rCalculating the probability of all possible moves.")
    actions = {}
    p = p.reshape(73, 8, 8)
    board = chess.Board(state)
    valid_moves = list(board.legal_moves)
    for moves in valid_moves:
        str_move = moves.uci()
        move = map_moves(str_move)
        actions[str_move] = p[move[0]][move[1]][move[2]]
    del board
    sys.stdout.write("\rCalculated the probability of all possible moves.")
    return actions

def map_moves(move : str) -> tuple:
    # Queen like moves
    move = move.lower()
    check : bool = len(move) == 5
    if check:
        check = check and (move[4].lower() != 'q')
        promotion = move[4].lower()
        move = move[:-1]
    move = [points[_]-1 if _ in points.keys() else (int(_) - 1) for _ in move]
    x = int(move[2] - move[0])
    y = int(move[3] - move[1])
    if check:
        if x==0 and promotion=='r':
            return (64 , move[0], move[1])
        if x==1 and promotion=='r':
            return (65 , move[0], move[1])
        if x==-1 and promotion=='r':
            return (66 , move[0], move[1])
        if x==0 and promotion=='b':
            return (67 , move[0], move[1])
        if x==1 and promotion=='b':
            return (68 , move[0], move[1])
        if x==-1 and promotion=='b':
            return (69 , move[0], move[1])
        if x==0 and promotion=='n':
            return (70 , move[0], move[1])
        if x==1 and promotion=='n':
            return (71 , move[0], move[1])
        if x==-1 and promotion=='n':
            return (72 , move[0], move[1])
    elif abs(x) == abs(y):
        if x>0 and y>0:
            return (x, move[0], move[1])
        elif x>0 and y<0:
            return (7+x, move[0], move[1])
        elif x<0 and y>0:
            return (14-x, move[0], move[1])
        elif x<0 and y<0:
            return (21-x, move[0], move[1])
    elif x==0 and y!=0:
        if y>0:
            return (28+y, move[0], move[1])
        else:
            return (35-y, move[0], move[1])
    elif x!=0 and y==0:
        if x>0:
            return (42+x, move[0], move[1])
        else:
            return (49-x, move[0], move[1])
    elif x==1 and y==2:
        return (56, move[0], move[1])
    elif x==2 and y==1:
        return (57, move[0], move[1])
    elif x==-1 and y==2:
        return (58, move[0], move[1])
    elif x==2 and y==-1:
        return (59, move[0], move[1])
    elif x==-2 and y==1:
        return (60, move[0], move[1])
    elif x==1 and y==-2:
        return (61, move[0], move[1])
    elif x==-1 and y==-2:
        return (62, move[0], move[1])
    elif x==-2 and y==-1:
        return (63, move[0], move[1])

def action_to_probs(root_node : Node) -> np.ndarray:
    sys.stdout.write("\rCalculating probabiliy of possible actions via MCTS algorithm.")
    prob_labels = np.zeros((73, 8, 8))
    N_ = root_node.N
    for edge in root_node.child_edges:
        move = map_moves(edge.action.uci())
        prob_labels[move[0]][move[1]][move[2]] = edge.N/N_
    sys.stdout.write("\rCalculated probabiliy of possible actions via MCTS algorithm.")
    return prob_labels
