from os import system
from tkinter import Variable
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

class LSTM(nn.Module):

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
        batchSize = x.shape[0]
        h_0 = torch.zeros(self.numLayer, batchSize, self.hiddenSize).to(device)
        c_0 = torch.zeros(self.numLayer, batchSize, self.hiddenSize).to(device)
        
        ## 第二个版本
        # Propagate input through LSTM
        seq_len = x.shape[1]
        features = x.shape[2]
        # x = x.view(self.batchSize, seq_len, features)
        ula, (h_out, _) = self.lstmLayer(x, (h_0, c_0))
        ula = ula.contiguous().view(batchSize * seq_len, self.hiddenSize)
        out = self.fc(ula)
        out = F.tanh(out)
        out = out.view(batchSize, seq_len, -1)
        out = out[:,-1,:]
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
        # x shape: batchsize, seqLen, features
        systemState = x[:,:, 0:-4]
        systemInput = x[:,:, -4:]

        h_0 = Variable(torch.zeros(self.numLayer, self.batchSize, self.hiddenSize).to(device))
        c_0 = Variable(torch.zeros(self.numLayer, self.batchSize, self.hiddenSize).to(device))

        seq_len = systemState.shape[1]  # 0 为batchsize
        features = systemState.shape[2]
        systemState = systemState.view(self.batchSize, seq_len, features)

        ula, (h_out, _) = self.lstmLayer(systemState, (h_0, c_0))
        # h_out = h_out.view(-1, self.hiddenSize)
        ula = ula.contiguous().view(self.batchSize * seq_len, self.hiddenSize)      ###???????
        lstm_out = self.lstmFullyConnector(ula)

        # integrate lstm out + system input
        y = torch.cat([lstm_out, systemInput, torch.ones(seq_len, 1).to(device)], dim=1)
        # fully connected layer output
        thetaY = self.fc(y)

        # dxdt = A*x + theta*
        systemState = systemState.view(systemState.shape[0], -1)
        out = self.a * systemState + thetaY
        # print(self.a)
        return out

class LSTM3(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM3, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2,output_size):
        super(ANN, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size, bias=True).to(device)
        self.layer2 = nn.Linear(hidden_size, hidden_size2, bias=True).to(device)
        self.layer3 = nn.Linear(hidden_size2, output_size, bias=True).to(device)

    def forward(self, x):   #开始计算的函数
        hidden = F.tanh(self.layer1(x))     #传入输入第一层
        # print("torch hidden", hidden)
        y_pred = F.tanh(self.layer2(hidden))       #传入输入第二层
        y_pred = self.layer3(y_pred)
        return y_pred

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features).to(device)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight)).to(device)
        if bias:
            self.bias = torch.Tensor(out_features).to(device)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias)).to(device)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nclass, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.graph_encoder1 = GCNLayer(nfeat, nhid).to(device)
        self.graph_encoder2 = GCNLayer(nhid, nhid2).to(device)
        self.graph_encoder3 = GCNLayer(nhid2, nclass).to(device)
        # self.fc1 = nn.Linear(nhid, nhid2).to(device)
        # self.fc2 = nn.Linear(nhid2, nclass).to(device)

    def forward(self, x, adj):
        x = self.graph_encoder1(x, adj)
        x = F.tanh(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.graph_encoder2(x, adj)
        x = F.tanh(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.graph_encoder3(x, adj)

        # x = self.fc1(x)
        # x = F.tanh(x)
        # x = self.fc2(x)
        return x

class GNN_LSTM(nn.Module):
    def __init__(self,inputSize, hiddenSize, 
                    outputSize, numLayer,
                    batchSize, bias=False):
        """
            inputSize: lstm 输入维度
            hiddenSize: lstm 隐藏层维度
            outputSize: lstm 输出维度
            numLayer: lstm 
        """
        super(GNN_LSTM, self).__init__()
        self.inputSize = inputSize
        self.numLayer = numLayer
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.bias = bias
        self.batchSize = batchSize
        # self.gcnLayer = GCNLayer(inputSize, inputSize)
        self.lstmLayer = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize,
                                num_layers=numLayer, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x, adj):
        # gcnOutput = self.gcnLayer(x, adj)
        x = torch.matmul(x, adj)

        batchSize = x.shape[0]
        h_0 = torch.zeros(self.numLayer, batchSize, self.hiddenSize).to(device)
        c_0 = torch.zeros(self.numLayer, batchSize, self.hiddenSize).to(device)
        seq_len = x.shape[1]
        features = x.shape[2]
        ula, (h_out, _) = self.lstmLayer(x, (h_0, c_0))
        ula = ula.contiguous().view(batchSize * seq_len, self.hiddenSize)
        out = self.fc(ula)
        out = F.tanh(out)
        out = out.view(batchSize, seq_len, -1)
        out = out[:,-1,:]
        return out

