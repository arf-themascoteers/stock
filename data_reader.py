import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

data = pd.read_csv("data/amazon.csv")
price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))


def split_data(stock, lookback):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    #data_raw = data_raw[:,0]

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size, :-1]
    y_train = data[:train_set_size, -1]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1]


    return [x_train, y_train, x_test, y_test]


def get_data():
    lookback = 20  # choose sequence length
    x_train, y_train, x_test, y_test = split_data(price, lookback)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    get_data()
