# Predicted model training file


import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel import LSTM, device, GNN_LSTM, LSTM3, ANN, GCN, normalize_sparse_adj
from Kinetics import cstr_cascading_kinetics
from scipy import sparse
torch.set_default_tensor_type(torch.DoubleTensor)
eps = 1e-50
adj = np.array([[0,1,0,0,0,0,1,0,0,0],
                [1,0,0,0,0,0,0,1,0,0],
                [1,1,0,0,0,0,0,0,0,0],
                [1,0,0,0,1,0,0,0,1,0],
                [0,1,0,1,0,0,0,0,0,1],
                [0,0,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])
adj = sparse.csc_matrix(adj)
adj = normalize_sparse_adj(adj)
adj = torch.Tensor(adj.todense()).to(device)

# parameters setting
Ea  = 50000     # activation energy J/gmol
R   = 8.314     # gas constant J/gmol/K
k0  = 8.46e6    # Arrhenius rate constant 1/min
V   = 1         # Volume [m3]
rho = 1000.0    # Density [kg/m3]
Cp  = 0.231     # Heat capacity [kJ/kg/K]
dHr = -1.15e4   # Enthalpy of reaction [kJ/kmol]
q   = 5.0       # Flowrate [m3/h]
cAi = 1.0       # Inlet feed concentration [mol/L]
Ti  = 350.0     # Inlet feed temperature [K]
cA0 = 0.5      # Initial concentration [mol/L]
T10 = 300    # Initial temperature of tank 1 [K]
T20 = 300     # Initial temperature of tank 2 [K]
kinetic1 = cstr_cascading_kinetics( Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)

eps = 1e-50
trainRate = 0.7
valRate = 0.2
testRate = 0.1

sampleInterval = 0.0001     # 采样时间 - 微分时间
loopNum = 100               # 采样周期

# generate data
#????????????????????
# data preprocess
windowsize = 5     

# read data
# inputData = np.load("data/windows/sontag/StateTraininputRecord.npy")
# outputData = np.load("data/windows/sontag/StateTrainoutputRecord_state.npy")

inputData = np.load("data/windows/sontag/StateTraininputRecord_rand.npy")
outputData = np.load("data/windows/sontag/StateTrainoutputRecord_state_rand.npy")


# 归一化操作
#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#
minInputValue = inputData.min(axis=1).min(axis=0)
maxInputValue = inputData.max(axis=1).max(axis=0)
minStateValue = minInputValue[0:6]
maxStateValue = maxInputValue[0:6]

minOutputValue = outputData.min(axis=1).min(axis=0)
maxOutputValue = outputData.max(axis=1).max(axis=0)

# 做归一化
inputData = (inputData - minInputValue) / (maxInputValue - minInputValue + eps)
outputData = (outputData - minOutputValue) / (maxOutputValue - minOutputValue + eps)

inputData = torch.Tensor(inputData).to(device)
outputData = torch.Tensor(outputData).to(device)

batchSize = 256
dataGroup = int(inputData.shape[0] / batchSize)
features = inputData.shape[2]
outputFeature = outputData.shape[2]
inputDataUse = torch.Tensor(np.zeros([dataGroup, batchSize, windowsize, features])).to(device)
outputDataUse = torch.Tensor(np.zeros([dataGroup, batchSize, outputFeature])).to(device)

for i in range(dataGroup):
    inputDataUse[i,:,:,:] = inputData[256*i:256*(i+1),:,:]
    outputDataUse[i,:,:] = outputData[256*i:256*(i+1),-1,:]

trainSizeNum = int(trainRate * dataGroup)
valSizeNum = int((trainRate+valRate) * dataGroup)

trainData = inputDataUse[0:trainSizeNum, :,:,:]
trainLable = outputDataUse[0:trainSizeNum, :,:]

valData = inputDataUse[trainSizeNum:valSizeNum, :,:,:]
valLabel = outputDataUse[trainSizeNum:valSizeNum, :,:]

testData = inputDataUse[valSizeNum:, :,:,:]
testLabel = outputDataUse[valSizeNum:, :,:]

## Train parameters
systemStateSize = features - 4
systemInput = 4
fullState = features + 1 # 多一个噪声项
hiddenSize = 64 # lstm 隐层个数
numlayer = 1    # lstm 层数
outputNum = outputFeature # 系统输出 - 状态量

num_epochs = 1800
learning_rate = 0.001

model = GNN_LSTM(inputSize=features, hiddenSize=hiddenSize,
        outputSize=outputNum, numLayer=numlayer,batchSize=batchSize).to(device)
# model = LSTM2(inputSize=systemStateSize, hiddenSize=hiddenSize, outputSize=outputNum, numLayer=numlayer, fullStateSize=fullState, batchSize=5000).to(device)
# model = LSTM3(outputNum, features, hiddenSize, numlayer).to(device)
# model = ANN(features, hiddenSize, 32, outputNum)

criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(0,num_epochs):
    if epoch % 10 == 0:
        print("This is the {} epoches".format(epoch))
    for t in range(0, trainData.shape[0]):
        trainTemp = trainData[t,:,:,:]
        outputs = model(trainTemp, adj)
        optimizer.zero_grad()
    # obtain the loss function
        trainLableTemp = trainLable[t,:,:]
    # trainLableTemp = trainLable
        loss = criterion(outputs, trainLableTemp)

        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        # 此处评价指标不对 需要修改 不应该是相加，而应该是将他们全部输出后存到数组中然后调用criterion进行判别
        trainLoss = 0
        for t in range(0, trainData.shape[0]):
            trainTemp = trainData[t,:,:]
            trainOutput = model(trainTemp, adj)
            trainLoss = trainLoss + criterion(trainOutput, trainLable[t,:,:])
        trainLoss = trainLoss / trainData.shape[0] 


        valLoss = 0
        for t in range(0, valData.shape[0]):
            valDataTemp = valData[t,:,:]
            valoutput = model(valDataTemp, adj)
            valLoss = valLoss + criterion(valoutput, valLabel[t,:,:])
        valLoss = valLoss / valData.shape[0] 
    
        print("Epoch: %d, loss: %1.8f, val loss: %1.8f" % (epoch, trainLoss.item(), valLoss.item()))

print("Training is over!")

# 测试阶段
model.eval()

# 保存网络
# PATH = "netModel/LSTM2_2_1_{}epoch.pth".format(num_epochs)
PATH = "netModel/GNN_LSTM_{}epoch.pth".format(num_epochs)
# torch.save(model, PATH)
torch.save({'model': model.state_dict()}, PATH)
# torch.save(model, PATH, _use_new_zipfile_serialization=False)

# predictedY = np.zeros([testData.shape[0], testLabel.shape[2]])
# realY = np.zeros([testData.shape[0],testLabel.shape[2]])
print("Test begin!!!!!!")   #@@@@########################## 测试使用机理模型进行验证？然后看下差距多大？
testLoss = 0

for t in range(0, testData.shape[0]):
    testDataTemp = testData[t,:,:,:]
    for i in range(testDataTemp.shape[0]):
        testDataUse = testDataTemp[i, :, :]
        testDataUse = testDataUse.unsqueeze(0)
        testPredict = model(testDataUse, adj)
        
        # testLoss = testLoss + criterion(testPredict, testLabel[t,:,:])

        #  first principle calculation
        testDatatempFp = testDataUse.squeeze().data.cpu().numpy()
        testDatatempFp = testDatatempFp[-1,:]
        testDataUsefp = testDatatempFp * (maxInputValue - minInputValue + eps) + minInputValue      
        state = testDataUsefp[0:6]
        control = testDataUsefp[6:]
        
        # fpOutput = kinetic1.systemDeriv(state, control)
        fpOutput = kinetic1.nextState(state, control, sampleInterval,loopNum)
        fpOutput = np.array(fpOutput)
        fpOutput = (fpOutput - minOutputValue) / (maxOutputValue - minOutputValue + eps)            # 归一化
        fpOutput = torch.Tensor(fpOutput).to(device).unsqueeze(0)
        testLoss = testLoss + criterion(testPredict, fpOutput)
        if (i == 1) and(t==1):
            print(testLoss)
            print(testPredict)
            print(fpOutput)


testLoss = testLoss / testData.shape[0]
print("Test results is:", testLoss)
print("Test End")

