# --utf-8--
# Data generation part
# The generated data is used for LSTM training

from matplotlib.pyplot import axis
import numpy as np
import math
import random

from Kinetics import *
from PredictionModel import *

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

controlInterval = 0.01      # 控制时间
sampleInterval = 0.005     # 采样时间 - 微分时间
loopNum = 1               # 采样周期

CA1 = 0.454
CA2 = 3.454
CB1 = cA0
CB2 = cA0
T1 = T10
T2 = T20

CA1s = 1.954
CA2s = 1.954
CA10s = 4.0
CA20s = 4.0

# generate  random input
uTemp = np.random.rand(10000, 4)  # delt_A10, delt_Q1, delt_A20, delt_Q2
u = uTemp * np.array([7, 1e6, 7, 1e6])
u = u + np.array([0.5, -5e5, 0.5, -5e5])


# generate obervation data with different input
# initial input
CA10 = cAi
CA20 = cAi
Q1 = 1e3
Q2 = 1e3
kinetic1 = cstr_cascading_kinetics( Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)
initialConditions = [CA1, T1, CB1, CA2, T2, CB2]
inputRecord = np.zeros([u.shape[0],10])  
outputRecord = np.zeros([u.shape[0],6])
outputRecord_state = np.zeros([u.shape[0],6])

t = 0
while(t < u.shape[0]):
    # print("This is the {} round loop".format(t))
    # if t % 100 == 0:
    #     initialConditions = np.random.rand(6) * np.array([3, 50, 3, 3, 50, 3]) + np.array([0.3, 300, 0.3, 0.3, 300, 0.3])
    
    # 记录状态
    tempState = np.zeros(10)
    tempState[0] = initialConditions[0]  # 当前状态
    tempState[1] = initialConditions[1]
    tempState[2] = initialConditions[2]
    tempState[3] = initialConditions[3] 
    tempState[4] = initialConditions[4]
    tempState[5] = initialConditions[5]  
    # 输入更新
    tempState[6] = u[t,0]   # 当前输入，第一级 入料
    tempState[7] = u[t,1]   # 当前输入，第一级 加热
    tempState[8] = u[t,2]   # 当前输入，第二级 入料
    tempState[9] = u[t,3]   # 当前输入，第二级 加热

    # 上一时刻的状态和输入得到当前观测量
    controlQuantity = [u[t,0], u[t,1], u[t,2], u[t,3]]
    variation = kinetic1.systemDeriv(initialConditions, controlQuantity)
    # state = kinetic1.nextState(initialState=initialConditions, controlQuantity=controlQuantity, timeInterval=sampleInterval, loopNum=loopNum)
    # 状态更新
    # initialConditions = state
    initialConditions = np.array(initialConditions) + np.array(variation) * sampleInterval

    if (np.isnan(variation).any() == False) and (np.isnan(initialConditions).any() == False):
        # 状态记录
        inputRecord[t,:] = tempState
        outputRecord[t,:] = np.array(variation)             # 记录变化量
        outputRecord_state[t,:] = np.array(initialConditions)  
        t = t + 1
    else:
        initialConditions = np.random.rand(6) * np.array([3, 50, 3, 3, 50, 3]) + np.array([0.3, 300, 0.3, 0.3, 300, 0.3])
        print("Relocated triggered!")
       

for i in range(inputRecord.shape[0]):
    for j in range(inputRecord.shape[1]):
        if (np.isnan(inputRecord[i,j])):
            print("NaN exist!")
            break
        
for i in range(outputRecord.shape[0]):
    for j in range(outputRecord.shape[1]):        
        if (np.isnan(outputRecord[i,j])):
            print("NaN exist in outputRecord")
            break
        #  之前可能是数据又问题，更换初始值，增加更换初始值步骤

np.save("data/variation/new_inputRecord_variation.npy", inputRecord)
np.save("data/variation/new_outputRecord_variation.npy", outputRecord)
np.save("data/variation/new_outputRecord_state_variation.npy", outputRecord_state)
print("Data generation success")