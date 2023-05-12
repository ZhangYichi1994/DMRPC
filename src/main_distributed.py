import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel import LSTM, device, LSTM2
# from pyomo.environ import * 
from Control import controlMethod
from controlMethod_all_approximateGrade import controlMethodApproximateGrade
from controlDecentralized import controlMethodDecentralized
from Kinetics import cstr_cascading_kinetics


c = controlMethodDecentralized()
c.choice = 'NN'
selfIterationFlag = False        # 是否使用自迭代选项：True 使用；  False 不实用
iterationTimeLimit = 3      # 分布式控制时间
controlInterval = 0.01      # 控制时间
sampleInterval = 0.0001     # 采样时间 - 微分时间
loopNum = 100               # 采样周期
eps = 1e-50

# parameters setting
Ea  = 50000     # activation energy kJ/gmol
R   = 8.314     # gas constant kJ/kmol/K
k0  = 8.46e6    # Arrhenius rate constant 1/min
V   = 1         # Volume [m3]
rho = 1000.0    # Density [kg/m3]
Cp  = 0.231     # Heat capacity [kJ/kg/K]
dHr = -1.15e4   # Enthalpy of reaction [kJ/kmol]
q   = 5.0       # Flowrate [m3/h]
cAi = 1.0       # Inlet feed concentration [mol/L]
Ti  = 350.0     # Inlet feed temperature [K]
cA0 = 0.5      # Initial concentration [mol/L]
T10 = 471.9    # Initial temperature of tank 1 [K]  +70
T20 = 331.9     # Initial temperature of tank 2 [K] -70

CA1 = 0.454
CA2 = 3.454
CB1 = cA0
CB2 = cA0
T1 = T10
T2 = T20

CA10 = 4
CA20 = 4
Q1 = 0
Q2 = 0

# 稳定点数值
CA1s = 1.954
CA2s = 1.954
CA10s = 4.0           # 控制稳定点
CA20s = 4.0
Q1s = 0.0
Q2s = 0.0
T1s = 401.9
T2s = 401.9

xs = np.array([CA1s, T1s, CA2s, T2s])    # 状态稳定点 —— 不稳定运行稳态
us = np.array([CA10s, Q1s, CA20s, Q2s])  # 控制稳定点 
initialConditions = np.array([CA1, T1, CB1, CA2, T2, CB2])
initialControl = np.array([CA10, Q1, CA20, Q2])

seqSize = 2         # MPC 控制序列长度
stepNum = 100
stateRecord = np.zeros([stepNum+1, 4])       # 记录系统状态，格式：[]
record = initialConditions[[0,1,3,4]] - c.xs
stateRecord[0, :] = record

# 准备LSTM的序列
lstmStateSeq = np.zeros([5, 10])
for loopTime in range(0, 5):
    controlSeq = np.array([0,0,0,0])
    controlUsed = controlSeq + c.us
    nowControls = np.array(controlUsed)
    nowConditions = np.array(initialConditions)             # 当前状态量
    nowInput = np.append(nowConditions, nowControls)
    newOutput = c.nextStep(nowInput, lstmStateSeq, method='FirstPrinciple')
    initialConditions = newOutput
    lstmStateSeq[loopTime, :] = nowInput
# 归一化，需要lstm 的输入序列是归一化之后的
lstmStateSeq = (lstmStateSeq - c.minInputValue) / (c.maxInputValue - c.minInputValue + eps)  
for loopTime in range(0,stepNum):
    if (selfIterationFlag == True) :
        pass
    elif (selfIterationFlag == False):
    # 根据运行状态，求解优化问题，计算控制序列
    # 辅助控制率计算
        controlSeqTemp1 = c.getAssistantControl(0, 0, 0, initialConditions,
                            0, lstmStateSeq, returnNum=0, modelType='NN') 
        controlSeqTemp2 = c.getAssistantControl(0, 0, 0, initialConditions,
                            0, lstmStateSeq, returnNum=1, modelType='NN') 
        for  times1 in range(iterationTimeLimit):        # 5次迭代，不知道能不能收敛
            # 第一个控制序列
            controlSeq1 = c.distributedLMPC(initialConditions, initialControl, seqSize, lstmStateSeq,0, controlSeqTemp2)
            controlSeq2 = c.distributedLMPC(initialConditions, initialControl, seqSize, lstmStateSeq,1, controlSeqTemp1)
            controlSeqTemp1 = controlSeq1 + np.zeros(controlSeq1.shape)
            controlSeqTemp2 = controlSeq2 + np.zeros(controlSeq2.shape)

        controlSeq = [controlSeq1[0], controlSeq1[1], controlSeq2[0], controlSeq2[1]]
        
        # 提取控制序列第一项，送入First-Principle模型或者LSTM，求解下一时刻系统状态
        controlUsed = controlSeq[0:4] + c.us          # 当前控制量
        nowControls = np.array(controlUsed)
        nowConditions = np.array(initialConditions)             # 当前状态量
        nowInput = np.append(nowConditions, nowControls)   
        newOutput = c.nextStep(nowInput, lstmStateSeq, method='FirstPrinciple')
        print("First principle output: ", newOutput)
        selfIteration = c.nextStep(nowInput, lstmStateSeq, method='NN')
        print("LSTM output is: ", selfIteration)
        initialConditions = newOutput            
        lstmStateSeq = c.updateInputSeq(lstmStateSeq, nowInput)  
        # 存数据， 前一时刻系统状态， 前一时刻控制输入， 当前时刻系统状态
        record = initialConditions[[0,1,3,4]] - c.xs
        stateRecord[loopTime+1, :] = record
    print("The {} step control is: {}, \n the state is: {}".format(loopTime, controlSeq[0:4], record))

timeTemp = range(0,stepNum+1)
print(timeTemp)
plt.subplot(4,1,1)
plt.plot(timeTemp, stateRecord[:,0])
plt.subplot(4,1,2)
plt.plot(timeTemp, stateRecord[:,1])
plt.subplot(4,1,3)
plt.plot(timeTemp, stateRecord[:,2])
plt.subplot(4,1,4)
plt.plot(timeTemp, stateRecord[:,3])
plt.savefig('test_LSTM_distributed.jpg')
plt.show()