import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
import data_reader
import os
import train

def test():
    model = Machine()
    if not os.path.isfile("models/machine.h5"):
        model = train.train()
        torch.save(model, "models/machine.h5")
    model = torch.load("models/machine.h5")
    x_train, y_train, x_test, y_test = data_reader.get_data()

    criterion = torch.nn.MSELoss(reduction='mean')
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test)
    print(f"Loss: {loss}")


if __name__ == "__main__":
    test()
