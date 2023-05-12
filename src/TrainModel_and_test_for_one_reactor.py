# Predicted model training file


import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel_for_one_reactor import LSTM, device, LSTM2
from tkinter import Variable
from sklearn.preprocessing import MinMaxScaler
eps = 1e-50

def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mae -- MAE 评价指标
    """
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae
 
def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mse -- MSE 评价指标
    """
    
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse

def slide_window(x, y, windowsize):
    datasize = x.shape[0] - windowsize + 1
    data = np.zeros([datasize, windowsize, x.shape[1]])
    label = np.zeros([datasize, windowsize, y.shape[1]])
    for i in range(0, datasize):
        temp = x[i:i+windowsize, :]
        data[i,:,:] = temp
        temp2 = y[i:i+windowsize, :]
        label[i,:,:] = temp2
    return data, label

# read data and normalization
inputData = np.load("data/inputRecord.npy")
outputData = np.load("data/outputRecord.npy")
sc = MinMaxScaler()

# 归一化操作
#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#
minInputValue = inputData.min(axis=0)
maxInputValue = inputData.max(axis=0)

minOutputValue = outputData.min(axis=0)
maxOutputValue = outputData.max(axis=0)

inputData = (inputData - minInputValue) / (maxInputValue - minInputValue + eps)
outputData = (outputData - minOutputValue) / (maxOutputValue - minOutputValue + eps)

# data preprocess
windowsize = 1     # 2个作为一组，搞一个batch???
inputData, outputData = slide_window(inputData, outputData, windowsize) # 滑窗了，说明有batch，没滑窗说明没有batch？？？？？？

inputData = torch.Tensor(inputData).to(device)
outputData = torch.Tensor(outputData).to(device)

trainData = inputData[0:5000, :]
trainLable = outputData[0:5000, :]

valData = inputData[5000:7000, :]
valLabel = outputData[5000:7000, :]

testData = inputData[7000:, :]
testLabel = outputData[7000:, :]

## Train parameters 只用第一个反应器的数据
features = 5
systemStateSize = 3
systemInput = 2
fullState = features + 1 # 多一个噪声项
hiddenSize = 32 # lstm 隐层个数
numlayer = 1    # lstm 层数
outputNum = 3 # 系统输出 - 状态量

num_epochs = 10
learning_rate = 0.001

# model = LSTM(inputSize=featureNum, hiddenSize=hiddenSize, outputSize=outputNum, numLayer=numLayer,batchSize=windowsize).to(device)
model = LSTM2(inputSize=systemStateSize, hiddenSize=hiddenSize, outputSize=outputNum, numLayer=numlayer, fullStateSize=fullState, batchSize=windowsize).to(device)

criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
listPicked = [0,1,2,6,7]
for epoch in range(0,num_epochs):
    print("This is the {} epoches".format(epoch))
    for t in range(0, trainData.shape[0]):
        # print("This is the {}-th sample".format(t))
        trainTemp = trainData[t,:,listPicked]
        outputs = model(trainTemp)
        optimizer.zero_grad()

        # obtain the loss function
        trainLableTemp = trainLable[t,:,0:3]
        loss = criterion(outputs, trainLableTemp)

        loss.backward()
        optimizer.step()
    
    # 此处评价指标不对 需要修改 不应该是相加，而应该是将他们全部输出后存到数组中然后调用criterion进行判别
    trainLoss = 0
    for t in range(0, trainData.shape[0]):
        trainTemp = trainData[t,:,listPicked]
        trainOutput = model(trainTemp)
        trainLoss = trainLoss + criterion(trainOutput, trainLable[t,:,0:3])
    trainLoss = trainLoss / trainData.shape[0]
    valLoss = 0
    for t in range(0, valData.shape[0]):
        valDataTemp = valData[t,:,listPicked]
        valoutput = model(valDataTemp)
        valLoss = valLoss + criterion(valoutput, valLabel[t,:,0:3])
    valLoss = valLoss / valData.shape[0]
    print("Epoch: %d, loss: %1.5f, val loss: %1.5f" % (epoch, trainLoss.item(), valLoss.item()))

print("Training is over!")

# 测试阶段
model.eval()
listPickedNew = [3,4,5,8,9]
predictedY = np.zeros([testData.shape[0], systemStateSize])
realY = np.zeros([testData.shape[0],systemStateSize])
for t in range(0, testData.shape[0]):
    testDataTemp = testData[t,:,listPicked]
    testPredict = model(testDataTemp)
    testPredict = testPredict[-1,:]
    dataPredict = testPredict.data.cpu().numpy()
    dataY_plot = testLabel[t,-1,3:6].data.cpu().numpy()

    # 存数组
    predictedY[t,:] = dataPredict
    realY[t,:] = dataY_plot

# 逆归一化
predictedY = predictedY * (maxOutputValue[0:3]- minOutputValue[0:3] + eps) + minOutputValue[0:3]
realY = realY * (maxOutputValue [0:3]- minOutputValue[0:3] + eps) + minOutputValue[0:3]

# 保存整个网络
PATH = "netModel/LSTM2_1_2_{}epoch.pth".format(num_epochs)
torch.save(model, PATH) 
# # 保存网络中的参数, 速度快，占空间少
# torch.save(model.state_dict(),PATH)

timeline = range(0,testData.shape[0])
plt.subplot(4,3,1)
plt.plot(timeline, predictedY[:,0])
plt.plot(timeline, realY[:,0])
plt.subplot(4,3,2)
plt.plot(timeline, predictedY[:,1])
plt.plot(timeline, realY[:,1])
plt.subplot(4,3,3)
plt.plot(timeline, predictedY[:,2])
plt.plot(timeline, realY[:,2])
# plt.subplot(4,3,4)
# plt.plot(timeline, predictedY[:,3])
# plt.plot(timeline, realY[:,3])
# plt.subplot(4,3,5)
# plt.plot(timeline, predictedY[:,4])
# plt.plot(timeline, realY[:,4])
# plt.subplot(4,3,6)
# plt.plot(timeline, predictedY[:,5])
# plt.plot(timeline, realY[:,5])
plt.savefig("test_for_one_reactor_{}_epoch_{}_windowSize.jpg".format(num_epochs, windowsize))
plt.show()
