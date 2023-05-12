from os import system
from tkinter import Variable
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    """LSTM layer reference from github:
    https://github.com/spdin/time-series-prediction-lstm-pytorch/blob/master/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb
    """
    def __init__(self, inputSize, hiddenSize, outputSize, numLayer, batchSize, bias=False):
        """
            inputSize: lstm 输入维度
            hiddenSize: lstm 隐藏层维度
            outputSize: lstm 输出维度
            numLayer: lstm 
        """
        super(LSTM, self).__init__()
        self.inputSize = inputSize
        self.numLayer = numLayer
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.bias = bias
        self.batchSize = batchSize

        self.lstmLayer = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize,
                                num_layers=numLayer, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)
        # self.hiddencell = (torch.zeros(self.numLayer, x.size(0), self.hiddenSize))

    def forward(self, x):
        h_0 = torch.zeros(self.numLayer, x.size(0), self.hiddenSize).to(device)
        c_0 = torch.randn(self.numLayer, x.size(0), self.hiddenSize).to(device)
        
        ## 第二个版本
        # Propagate input through LSTM
        seq_len = x.shape[0]
        features = x.shape[1]
        x = x.view(self.batchSize, 1, features)
        ula, (h_out, _) = self.lstmLayer(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hiddenSize)

        # out = F.tanh(h_out)

        out = self.fc(h_out)

        # out = F.ReLu(out)

        return out

class LSTM2(nn.Module):
    """LSTM layer reference from github:
    https://github.com/spdin/time-series-prediction-lstm-pytorch/blob/master/Time_Series_Prediction_with_LSTM_Using_PyTorch.ipynb
    """
    def __init__(self, inputSize, hiddenSize, outputSize, numLayer, fullStateSize, batchSize, bias=False):
        """
            inputSize: lstm 输入维度
            hiddenSize: lstm 隐藏层维度
            outputSize: lstm 输出维度
            numLayer: lstm 内部层数
            fullState: 全连接层维度
        """
        super(LSTM2, self).__init__()
        self.inputSize = inputSize          # lstm input size
        self.numLayer = numLayer            # lstm layer num
        self.hiddenSize = hiddenSize        # lstm hidden size 
        self.outputSize = outputSize        # model predicted size
        self.bias = bias
        self.fullState = fullStateSize      # model state size = systemState + systemInput
        self.batchSize = batchSize

        self.lstmLayer = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize,
                                num_layers=numLayer, batch_first=True)
        self.lstmFullyConnector = nn.Linear(hiddenSize, self.inputSize)
        self.fc = nn.Linear(fullStateSize, outputSize)
        self.a = Variable(torch.randn(1, inputSize)).to(device)

    def forward(self, x):
        systemState = x[:, 0:-2]
        systemInput = x[:, -2:]

        h_0 = Variable(torch.zeros(self.numLayer, self.batchSize, self.hiddenSize).to(device))
        c_0 = Variable(torch.zeros(self.numLayer, self.batchSize, self.hiddenSize).to(device))

        seq_len = systemState.shape[0]
        features = systemState.shape[1]
        systemState = systemState.view(self.batchSize, 1, features)

        ula, (h_out, _) = self.lstmLayer(systemState, (h_0, c_0))
        h_out = h_out.view(-1, self.hiddenSize)
        lstm_out = self.lstmFullyConnector(h_out)

        # integrate lstm out + system input
        y = torch.cat([lstm_out, systemInput, torch.ones(seq_len, 1).to(device)], dim=1)
        # fully connected layer output
        thetaY = self.fc(y)

        # dxdt = A*x + theta*
        systemState = systemState.view(systemState.shape[0], -1)
        out = self.a * systemState + thetaY
        # print(self.a)
        return out