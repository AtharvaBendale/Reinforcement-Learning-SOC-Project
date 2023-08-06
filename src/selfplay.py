from model import CustomModel
import os
import config
import torch
from agent import Agent

def selfplay():
    model = CustomModel()

    if os.path.exists("./model.pth"):
        model.load_state_dict(torch.load("./model.pth"))
    else:
        torch.save(model.state_dict(), "./model.pth")

    for _ in range(1, config.SELFPLAY_BATCH_SIZE+1):
        agent = Agent(model, config.EPSILON, _)
        del agent