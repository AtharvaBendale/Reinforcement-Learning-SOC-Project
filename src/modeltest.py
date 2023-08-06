from model import CustomModel
import torch
import numpy as np
model = CustomModel()
model.load_state_dict(torch.load("./model.pth"))

for _ in range(15):

    y = model.forward(np.random.random(size=(19, 8, 8)))
    policy, value = np.array(y[0].detach()).reshape(73, 8, 8), np.array(y[1].detach())[0][0]
    print(policy)
    assert not np.all(policy ==0)