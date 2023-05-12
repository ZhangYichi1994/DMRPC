# 问题： 与采样周期和控制周期有关，太小了还不行
# 梯度计算有问题，梯度的计算不能按照我这个方法进行，近似梯度必须用微分方程来计算

import numpy as np
import matplotlib.pyplot as plt
from controlMethod_all_approximateGrade import controlMethodApproximateGrade
from scipy.optimize import minimize

def attackProcess(initialState, attackStartTime, loopTime,
                  stateRecord, attackType):

    stateRecordUse = stateRecord[0:attackStartTime, :]
    initialState = np.array(initialState)
    stateUse = initialState[[0,1,3,4]]
    rho_x = 10.0
    bound = []
    bound.append((-3.5, 3.5))
    bound.append((-5e5, 5e5))

    P = np.array([[1060.0, 22.0],[22.0, 0.52]])  # size: 4x4
    def calLy(x):
        x = np.array(x)
        return -x.dot(P).dot(x)
    def calCon(x):
        x = np.array(x)
        return rho_x - x.dot(P).dot(x)
    xInitial = [0.0, 0.0]

    if attackType == 'MinMax':
        cons = {'type': 'ineq', 'fun' : calCon}
        res = minimize(calLy, xInitial, bounds=bound, method='SLSQP',constraints=cons, tol = 1e-10)
        attackValue = res.x

    if attackType == 'Geo':
        beta = 0.1
        alpha = 0.5
        initialState = initialState + \
                    beta * np.power((1+alpha), loopTime - attackStartTime)
        attackValue = initialState[[3,4]]
    
    if attackType == 'Surge':
        time2Change = 5
        cons = {'type': 'ineq', 'fun' : calCon}
        res = minimize(calLy, xInitial, bounds=bound, method='SLSQP',constraints=cons, tol = 1e-10)
        attackValue = res.x
        if (loopTime - attackStartTime) >= time2Change:
            stateRecordUse = np.abs(stateRecordUse)
            stateRecordUseMax = np.max(stateRecordUse, axis=1)
            attackvalue = stateRecordUseMax[[3,4]]
    return attackValue

# c = controlMethod()
c = controlMethodApproximateGrade()
# c.choice = 'FirstPrinciple'
c.choice = 'NN'
selfIterationFlag = True        # 是否使用自迭代选项：True 使用；  False 不使用
attackType = 'Surge'              # 'MinMax', 'Surge' , 'Geo', 
detectionFlag = True
rollBackWinSize = 3

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


controlInterval = 0.01      # 控制时间
sampleInterval = 0.0001     # 采样时间 - 微分时间
loopNum = 100               # 采样周期
eps = 1e-50

seqSize = 2         # MPC 控制序列长度
stepNum = 100
stateRecord = np.zeros([stepNum+1, 4])       # 记录系统状态，格式：[]
record = initialConditions[[0,1,3,4]] - c.xs
stateRecord[0, :] = record
diffRecord = np.zeros([stepNum + 1, 1])

# 准备LSTM的序列
lstmStateSeq = np.zeros([5, 10])
lstmStateSeq_backup = np.zeros([20,10])     # LSTM状态序列备份， 用于攻击发现后的回滚
for loopTime in range(0, 5):
    controlSeq = np.array([0,0,0,0])
    controlUsed = controlSeq + c.us
    nowControls = np.array(controlUsed)
    nowConditions = np.array(initialConditions)             # 当前状态量
    nowInput = np.append(nowConditions, nowControls)
    newOutput = c.nextStep(nowInput, lstmStateSeq, method='FirstPrinciple')
    initialConditions = newOutput
    lstmStateSeq[loopTime, :] = nowInput
    initialConditionsForControl = np.array(initialConditions)
# 归一化，需要lstm 的输入序列是归一化之后的
lstmStateSeq = (lstmStateSeq - c.minInputValue) / (c.maxInputValue - c.minInputValue + eps)  
lstmStateSeq_backup[-5:, :] = lstmStateSeq 

attackFlag = False
attackStartTime = 1000
attackStopTime = 60
rollBackFlagAlready = True  # 判断数据是否已经回滚，已经回滚则不需要多次进行回滚操作，只需要平替就行
rollBackFlag = False        # 判断是否执行数据回滚操作
for loopTime in range(0,stepNum):
    if loopTime >= attackStartTime:
        attackFlag = True
    if loopTime >= attackStopTime:
        attackFlag = False
    if (selfIterationFlag == True) :
        # 根据运行状态，求解优化问题，计算控制序列
        controlSeq = c.CentralizeLMPC(initialConditions, initialControl, seqSize, lstmStateSeq)
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
        initialConditionsForControl = np.array(initialConditions)
        diffBetLSTM_and_FP = np.linalg.norm(initialConditions-selfIteration, 2)

        # 存数据， 前一时刻系统状态， 前一时刻控制输入， 当前时刻系统状态
        record = initialConditions[[0,1,3,4]] - c.xs
        stateRecord[loopTime+1, :] = record

        # 自迭代结果
        selfIterationOuput = c.nextStep(nowInput, lstmStateSeq, method='NN')
        initialConditions = selfIterationOuput
        lstmStateSeq = c.updateInputSeq(lstmStateSeq, nowInput)
        lstmStateSeq_backup = c.updateInputSeq(lstmStateSeq_backup, nowInput)

    elif (selfIterationFlag == False):
        # 根据运行状态，求解优化问题，计算控制序列
        controlSeq = c.CentralizeLMPC(initialConditionsForControl, initialControl, seqSize, lstmStateSeq)
        # 提取控制序列第一项，送入First-Principle模型或者LSTM，求解下一时刻系统状态
        controlUsed = controlSeq[0:4] + c.us          # 当前控制量
        print("controlUsed is: ", controlUsed)
        nowControls = np.array(controlUsed)
        nowConditions = np.array(initialConditions)             # 当前状态量
        nowInput = np.append(nowConditions, nowControls)   
        nowInputForLSTM = np.append(initialConditionsForControl, nowControls)   # 喂给LSTM需要更新的序列，传感器内容为攻击后的值
        print("nowInputForLSTM is: ", nowInputForLSTM)
        newOutput = c.nextStep(nowInput, lstmStateSeq, method='FirstPrinciple')
        print("First principle output: ", newOutput)
        selfIteration = c.nextStep(nowInputForLSTM, lstmStateSeq, method='NN')
        print("LSTM output is: ", selfIteration)
        initialConditions = newOutput
        initialConditionsForControl = np.array(initialConditions)
        diffBetLSTM_and_FP = np.linalg.norm(initialConditions-selfIteration, 2)
        # print("The difference between LSTM and FP is:", np.linalg.norm(initialConditions - selfIteration, 2))
        
       
        # 存数据， 前一时刻系统状态， 前一时刻控制输入， 当前时刻系统状态
        record = initialConditions[[0,1,3,4]] - c.xs
        stateRecord[loopTime+1, :] = record

        # 更新LSTM序列
        lstmStateSeq = c.updateInputSeq(lstmStateSeq, nowInputForLSTM)
        lstmStateSeq_backup = c.updateInputSeq(lstmStateSeq_backup, nowInputForLSTM)
        
    diffRecord[loopTime + 1, :] = diffBetLSTM_and_FP

    print("The {} step control is: {}, \n the state is: {}".format(loopTime, controlSeq[0:4], record))

timeTemp = range(0,stepNum+1)
print(timeTemp)
plt.subplot(5,1,1)
plt.plot(timeTemp, stateRecord[:,0])
plt.subplot(5,1,2)
plt.plot(timeTemp, stateRecord[:,1])
plt.subplot(5,1,3)
plt.plot(timeTemp, stateRecord[:,2])
plt.subplot(5,1,4)
plt.plot(timeTemp, stateRecord[:,3])
plt.subplot(5,1,5)
plt.plot(timeTemp, diffRecord[:,0])
plt.savefig('test_attack{}_{}_detect_rb.jpg'.format(attackType, detectionFlag))
plt.show()