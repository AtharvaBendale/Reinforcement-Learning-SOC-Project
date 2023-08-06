from model import CustomModel
import torch
import numpy as np
import math

# Loading the saved model for training
def train(num_files : int , base_path : str):
    model = CustomModel()
    model.load_state_dict(torch.load("./model.pth"))

    # Initialising data containers
    prob_array = np.load(base_path+"/game_1/prob_label.npy")
    win_array = np.ones((prob_array.shape[0], 1))*int(np.load(base_path+"/game_1/win.npy"))
    board_array = np.load(base_path+"/game_1/board.npy")

    # Loading the training data
    for _ in range(2, num_files+1):
        buf_prob_array = np.load(base_path+f"/game_{_}/prob_label.npy")
        buf_win_array = np.ones((buf_prob_array.shape[0], 1))*int(np.load(base_path+f"/game_{_}/win.npy"))
        buf_board_array = np.load(base_path+f"/game_{_}/board.npy")
        prob_array = np.append(prob_array, buf_prob_array, axis=0)
        win_array = np.append(win_array, buf_win_array, axis=0)
        board_array = np.append(board_array, buf_board_array, axis=0)
 
    assert board_array.shape[1:] == (19, 8, 8)
    assert win_array.shape[1] == 1
    assert win_array.shape[0] == board_array.shape[0]
    assert prob_array.shape[1:] == (73, 8, 8)

    # Training
    model.train_model(board_array, prob_array, win_array, 10, math.ceil(num_files/4) if num_files <32 else 16, 0.005)
    torch.save(model.state_dict(), "./model.pth")
    print("Model training finished!")