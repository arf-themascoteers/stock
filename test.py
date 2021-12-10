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


    y_test_pred = model(x_test)
    print(y_test_pred[10:15])
    print(y_test[10:15])


if __name__ == "__main__":
    test()
