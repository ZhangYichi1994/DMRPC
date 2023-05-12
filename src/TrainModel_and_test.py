# Predicted model training file


import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel import LSTM, device, LSTM2, LSTM3, ANN, GCN
from tkinter import Variable
# from sklearn.preprocessing import MinMaxScaler
eps = 1e-50
adj = np.array([[1,1,0,0,0,0,1,0,0,0],
                [1,1,0,0,0,0,0,1,0,0],
                [1,1,1,0,0,0,0,0,0,0],
                [1,0,0,1,1,0,0,0,1,0],
                [0,1,0,1,1,0,0,0,0,1],
                [0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])
adj = torch.Tensor(adj).to(device)

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
inputData = np.load("data/variation/inputRecord_variation.npy")
outputData = np.load("data/variation/outputRecord_variation.npy")
outputData_state = np.load("data/variation/outputRecord_state_variation.npy")

# inputData = np.load("data/inputRecord.npy")
# outputData = np.load("data/outputRecord.npy")

# 归一化操作
#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#
minInputValue = inputData.min(axis=0)
maxInputValue = inputData.max(axis=0)
minStateValue = minInputValue[0:6]
maxStateValue = maxInputValue[0:6]

minOutputValue = outputData.min(axis=0)
maxOutputValue = outputData.max(axis=0)

# minOutput_stateValue = outputData_state.min(axis=0)
# maxOutput_stateValue = outputData_state.max(axis=0)

# 做归一化
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

## Train parameters
features = trainData.shape[2]
systemStateSize = features - 4
systemInput = 4
fullState = features + 1 # 多一个噪声项
hiddenSize = 64 # lstm 隐层个数
numlayer = 1    # lstm 层数
outputNum = 6 # 系统输出 - 状态量

num_epochs = 50
learning_rate = 0.001

# model = LSTM(inputSize=featureNum, hiddenSize=hiddenSize, outputSize=outputNum, numLayer=numLayer,batchSize=windowsize).to(device)
# model = LSTM2(inputSize=systemStateSize, hiddenSize=hiddenSize, outputSize=outputNum, numLayer=numlayer, fullStateSize=fullState, batchSize=windowsize).to(device)
# model = LSTM3(outputNum, features, hiddenSize, numlayer).to(device)
# model = ANN(features, hiddenSize, 32, outputNum)
model = GCN(1, 32, 16, 1, 0.5)

criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(0,num_epochs):
    print("This is the {} epoches".format(epoch))
    for t in range(0, trainData.shape[0]):
        # print("This is the {}-th sample".format(t))
        trainTemp = trainData[t,:,:]
        # trainTemp = trainData
        # trainTemp = trainTemp.unsqueeze(0)
        # outputs = model(trainTemp)
        outputs = model(trainTemp.T, adj)
        optimizer.zero_grad()

        # obtain the loss function
        trainLableTemp = trainLable[t,:,:]
        # trainLableTemp = trainLable
        loss = criterion(outputs[0:6,:].T, trainLableTemp)

        loss.backward()
        optimizer.step()
    
    # 此处评价指标不对 需要修改 不应该是相加，而应该是将他们全部输出后存到数组中然后调用criterion进行判别
    trainLoss = 0
    for t in range(0, trainData.shape[0]):
        trainTemp = trainData[t,:,:]
        trainOutput = model(trainTemp.T, adj)
        trainLoss = trainLoss + criterion(trainOutput[0:6].T, trainLable[t,:,:])
    trainLoss = trainLoss / trainData.shape[0]


    valLoss = 0
    for t in range(0, valData.shape[0]):
        valDataTemp = valData[t,:,:]
        valoutput = model(valDataTemp.T, adj)
        valLoss = valLoss + criterion(valoutput[0:6].T, valLabel[t,:,:])
    # valOutput = model(valData)
    # valLoss = criterion(valOutput, valLabel)
    valLoss = valLoss / valData.shape[0]
    print("Epoch: %d, loss: %1.5f, val loss: %1.5f" % (epoch, trainLoss.item(), valLoss.item()))

print("Training is over!")

# 测试阶段
model.eval()

# 保存网络
# PATH = "netModel/LSTM2_2_1_{}epoch.pth".format(num_epochs)
PATH = "netModel/GCN_variation.pth"
# torch.save(model, PATH)
torch.save({'model': model.state_dict()}, PATH)
# torch.save(model, PATH, _use_new_zipfile_serialization=False)

predictedY = np.zeros([testData.shape[0], testLabel.shape[2]])
realY = np.zeros([testData.shape[0],testLabel.shape[2]])

# for t in range(0, testData.shape[0]):
#     testDataTemp = testData[t,:,:]
#     testPredict = model(testDataTemp)
#     testPredict = testPredict[-1,:]
#     dataPredict = testPredict.data.cpu().numpy()
#     dataY_plot = testLabel[t,-1,:].data.cpu().numpy()

#     # 存数组
#     predictedY[t,:] = dataPredict
#     realY[t,:] = dataY_plot

# # 逆归一化
# predictedY = predictedY * (maxOutputValue - minOutputValue + eps) + minOutputValue
# realY = realY * (maxOutputValue - minOutputValue + eps) + minOutputValue

print("Test begin!!!!!!")
loopNum = 1
sampleInterval = 0.005
for t in range(0, testData.shape[0]):
    testDataTemp = testData[t,:,:]
    controlNow = testDataTemp[:,6:]
    stateNow = testDataTemp[:,0:6]
    controlNow = controlNow.data.cpu().numpy()
    stateNow = stateNow.data.cpu().numpy()      # 扔回numpy
    # stateNow 逆归一化
    stateNow = stateNow * (maxStateValue - minStateValue + eps) + minStateValue
    
    state_and_control = testDataTemp
    newStateVar = model(state_and_control.T,adj)
    newStateVar = newStateVar[-1,:]
    newStateVar = newStateVar.data.cpu().numpy()
    newStateVar = newStateVar * (maxOutputValue - minOutputValue + eps) + minOutputValue
    
    stateNow = stateNow + newStateVar * sampleInterval

    # stateNow = (stateNow - minStateValue) / (maxStateValue - minStateValue + eps)
    # state_and_control = np.append(stateNow, controlNow)
    # state_and_control = torch.tensor(state_and_control).to(device)
    # state_and_control = state_and_control.unsqueeze(0)

    predictedY[t,:] = stateNow
    # realY[t,:] = testLabel[t,-1,:].data.cpu().numpy()
    realY[t,:] = outputData_state[7000 + t, :]

# predictedY = predictedY * (maxOutputValue - minOutputValue + eps) + minOutputValue
# realY = realY * (maxOutputValue - minOutputValue + eps) + minOutputValue
print("Test End")

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
plt.subplot(4,3,4)
plt.plot(timeline, predictedY[:,3])
plt.plot(timeline, realY[:,3])
plt.subplot(4,3,5)
plt.plot(timeline, predictedY[:,4])
plt.plot(timeline, realY[:,4])
plt.subplot(4,3,6)
plt.plot(timeline, predictedY[:,5])
plt.plot(timeline, realY[:,5])
plt.savefig('test_GCN_var.jpg')
# plt.show()
print("Finished!")

# 逆归一化
# predictedY = scalarOutput.inverse_transform(predictedY)
# realY = scalarOutput.inverse_transform(realY)

# # 保存整个网络
# PATH = "netModel/LSTM2/"
# torch.save(model, PATH) 
# 保存网络中的参数, 速度快，占空间少
# torch.save(model.state_dict(),PATH)
#--------------------------------------------------
#针对上面一般的保存方法，加载的方法分别是：
# model_dict=torch.load(PATH)
# model_dict=model.load_state_dict(torch.load(PATH))

# 画图 - 
# sizeLength = int(len(outputData) * 0.67)
# timeline = np.linspace(0, testData.shape[0])
# plt.plot(timeline, dataY_plot, label='Real')
# plt.figure()
# plt.plot(timeline, dataPredict, label='Predict')
# plt.suptitle('Time-Series Prediction')
# plt.show()

