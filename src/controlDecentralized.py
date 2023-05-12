from secrets import choice
from sys import set_coroutine_origin_tracking_depth
import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel import LSTM, device, LSTM2
# from pyomo.environ import * 
from Kinetics import cstr_cascading_kinetics
from scipy.optimize import minimize, NonlinearConstraint
torch.set_default_tensor_type(torch.DoubleTensor)


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
T10 = 300.0    # Initial temperature of tank 1 [K]
T20 = 300.0     # Initial temperature of tank 2 [K]

controlInterval = 0.01      # 控制时间
sampleInterval = 0.0001     # 采样时间 - 微分时间
loopNum = 100     
eps = 1e-50   
class controlMethodDecentralized():
    def __init__(self):
        # 稳定点数值
        self.CA1s = 1.954
        self.CA2s = 1.954
        self.CA10s = 4.0
        self.CA20s = 4.0
        self.Q1s = 0.0
        self.Q2s = 0.0
        self.T1s = 401.9
        self.T2s = 401.9
        self.rhonn = 10.0
        self.rhomin = 12.0
        self.rho_1_2 = 380.0
        self.xs = np.array([self.CA1s, self.T1s, self.CA2s, self.T2s])   # 稳定点     ???????????????
        self.us = np.array([self.CA10s, self.Q1s, self.CA20s, self.Q2s])
        self.P = np.array([[1060.0, 22.0],[22.0, 0.52]])  # size: 4x4
        self.P2 = np.array([[1060.0, 22.0, 0.0, 0.0],     # size: 8x8
                            [22.0, 0.52, 0.0, 0.0],
                            [0.0, 0.0, 1060.0, 22.0],
                            [0.0, 0.0, 22.0, 0.52]])
        self.Q = np.diag([2e3, 1.0, 2e3, 1.0])      # size： 4x4
        self.Q2Matrix = np.diag([2e3, 1.0])               # size: 2x2
        # self.R = np.diag([8e-13, 0.001])        # size：2x2
        self.R = np.diag([0.001, 8e-13])
        self.PATH = "netModel/LSTM_simple_outRe_state_1800epoch.pth"
        self.model = LSTM(10, 64, 6, 1, 256).to(device)
        self.model.load_state_dict(torch.load(self.PATH)['model'])
        # self.choice = 'FirstPrinciple'    # or  'NN'   'FirstPrinciple'
        self.choice = 'NN'
        self.fpModel = cstr_cascading_kinetics(Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)

        # for ipyopt used
        self.seqSize = 5
        self.initialConditions = np.zeros(1)
        self.initialControls = np.zeros(1)
        self.minInputValue, self.maxInputValue, self.minOutputValue, self.maxOutputValue = self.getRegularItem()

    def updateInputSeq(self, oldInput, newInput):
        # 更新LSTM的输入序列
        # 归一化
        # 输入 输出 均为 numpy类型
        newInput = (newInput - self.minInputValue) / (self.maxInputValue - self.minInputValue + eps)
        newSeq = np.zeros(oldInput.shape)
        newSeq[0:-1,:] = oldInput[1:,:]
        newSeq[-1,:] = newInput
        return newSeq

    def rollBack(self, lstmSeq, windowSize):
        length = lstmSeq.shape[0]
        if windowSize < (length - 5):             # 这个 5 与lstm的时间窗口长度有关
            controlSeq = lstmSeq[:,6:]
            newLSTMSeq = lstmSeq[-10:-5,:]
            for i in range(windowSize):
                newLSTMSeq = torch.Tensor(newLSTMSeq).to(device)
                newLSTMSeq = newLSTMSeq.unsqueeze(0)
                output = self.model(newLSTMSeq)
                output = output.data.cpu().numpy()
                output = np.squeeze(output)
                updated = np.append(output, controlSeq[i,:])
                newLSTMSeq = newLSTMSeq.data.cpu().numpy()
                newLSTMSeq = self.updateInputSeq(newLSTMSeq, updated)
            lstmHistory = lstmSeq
            lstmHistory[-5:,:] = newLSTMSeq

            newLSTMSeq = torch.Tensor(newLSTMSeq).to(device)
            newLSTMSeq = newLSTMSeq.unsqueeze(0)
            output = self.model(newLSTMSeq)
            output = output.data.cpu().numpy()
            output = np.squeeze(output)
            output = output * (self.maxOutputValue - self.minOutputValue + eps) + self.minOutputValue
            newLSTMSeq = newLSTMSeq.data.cpu().numpy()
            return lstmHistory, newLSTMSeq, output
        else:
            print("Window size longer than the sequence length")
            return np.NaN, np.NaN, np.NaN

    def modelNextState(self, initialSeq, inputState):
        # 输出已经 逆归一化 之后的结果
        initialSeq = self.updateInputSeq(initialSeq, inputState)
        stateUse = torch.Tensor(initialSeq).to(device)
        stateUse = stateUse.unsqueeze(0)
        predict = self.model(stateUse)
        predict = predict.data.cpu().numpy()
        predict = predict * (self.maxOutputValue - self.minOutputValue + eps) + self.minOutputValue
        predict = np.squeeze(predict)
        return predict

    def getOneStepGrad(self, initialConditions, controlQuantity, initialLSTMSeq, method):
        if method == 'NN':
            initialConditions = np.array(initialConditions)
            controlQuantity = np.array(controlQuantity)
            state_and_control = np.append(initialConditions,controlQuantity)
            # 归一化
            newState = self.modelNextState(initialLSTMSeq, state_and_control)
            stateVar = (newState - initialConditions) / (loopNum * sampleInterval)
        elif method == 'FirstPrinciple':
            stateVar = np.array(self.fpModel.systemDeriv(initialConditions, controlQuantity))
        else:
            print("wrong gradient method!!!")
        return stateVar

    def getRegularItem(self):
        # read data and normalization
        inputData = np.load("data/windows/sontag/StateTraininputRecord_rand.npy")
        outputData = np.load("data/windows/sontag/StateTrainoutputRecord_state_rand.npy")
        minInputValue = inputData.min(axis=1).min(axis=0)
        maxInputValue = inputData.max(axis=1).max(axis=0)
        minOutputValue = outputData.min(axis=1).min(axis=0)
        maxOutputValue = outputData.max(axis=1).max(axis=0)
        return minInputValue, maxInputValue, minOutputValue, maxOutputValue

    def nextStep(self, inputVar, initialLSTMSeq, method='NN'):
        # initialLSTMSeq 要求为 归一化后的矩阵
        if method == 'NN':
            # initialConditions = np.array(inputVar[0:6])
            # controlQuantity = np.array(inputVar[6:])
            # 送入神经网络的输入进行归一化处理
            # state_and_control = self.model.updateInputSeq(initialLSTMSeq, inputVar)
            newState = self.modelNextState(initialLSTMSeq, inputVar)
            output = newState
        
        elif method == 'FirstPrinciple':
            initialConditions = inputVar[0:6]
            controlQuantity = inputVar[6:]
            output = self.fpModel.nextState(initialState=initialConditions, 
                                controlQuantity=controlQuantity, 
                                timeInterval=sampleInterval, loopNum=loopNum)
        return np.array(output)

    def calObjective(self, controlSeq, seqSize, initialConditions, 
                    initialControls, initialLSTMSeq, controlNum, anotherContrlSeq):
        # controlSeq, 2个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
        # anotherControlSeq 另一个控制器的控制序列,2个一组，与controlSeq大小相同
        sum = 0

        # # 辅助控制率和要优化的控制率拼接成一个控制序列
        length = len(controlSeq)
        controlSeq = np.array(controlSeq)
        controlSeq = controlSeq.reshape(seqSize, int(length/seqSize))
        anotherContrlSeq = np.array(anotherContrlSeq)
        anotherContrlSeq = anotherContrlSeq.reshape(seqSize, int(length/seqSize))        
        if controlNum == 0:
            controlSeq = np.hstack((controlSeq, anotherContrlSeq))
        elif controlNum == 1:
            controlSeq = np.hstack((anotherContrlSeq, controlSeq))
        else:
            print("Wrong Number calConstraints")
        controlSeq = controlSeq.flatten()

        for i in range(0, seqSize):
            # u需要分情况进行拼接成四个来送到模型中进行计算
            # 第一个控制器和第二个控制器分别来进行计算
            # 要考虑是什么类型的，是顺序式的还是迭代式的
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
            nowControls = self.us + deltU           # 当前控制量
            u1Temp = nowControls[[0,1]]
            u2Temp = nowControls[[2,3]]
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice)
            initialConditions = newOutput
            initialLSTMSeq = self.updateInputSeq(initialLSTMSeq, nowInput)
            xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            xTemp1 = xTemp[[0,1]]
            xTemp2 = xTemp[[2,3]]
            if controlNum == 0:
                sum = sum + xTemp1.T.dot(self.Q2Matrix).dot(xTemp1) + \
                        xTemp2.T.dot(self.Q2Matrix).dot(xTemp2) + \
                        u1Temp.T.dot(self.R).dot(u1Temp)
            elif controlNum == 1:
                sum = sum + xTemp2.T.dot(self.Q2Matrix).dot(xTemp2) + \
                        xTemp1.T.dot(self.Q2Matrix).dot(xTemp1) + \
                        u2Temp.T.dot(self.R).dot(u2Temp)
            else:
                print("Wrong Number!!!")
        return sum
    def getObjJac(self, controlSeq, seqSize, initialConditions, 
                 initialControls, initialLSTMSeq,controlNum, anotherContrlSeq):
        initialConditions = np.array(initialConditions)
        controlSeq = np.array(controlSeq)
        stepLen = sampleInterval * loopNum      # 求微分时候的步长、
        jacVec = np.zeros(controlSeq.shape)     # 约束导数的大小
        seqLen = controlSeq.shape[0]

        oringinPoint = self.calObjective(controlSeq, seqSize, initialConditions,
                                        initialControls, initialLSTMSeq,
                                        controlNum, anotherContrlSeq)
        for j in range(seqLen):
            if j % 2 == 0:
                stepLenUse = stepLen
            else:
                stepLenUse = stepLen * 1
            controlPosition = j
            controlSeqNew = np.zeros(controlSeq.shape) + controlSeq
            controlSeqNew[controlPosition] = controlSeq[controlPosition] + \
                                                    stepLenUse
            originPointNow = self.calObjective(controlSeqNew, seqSize,
                                                initialConditions,
                                                initialControls, initialLSTMSeq,
                                                controlNum, anotherContrlSeq)
            jacVec[j] = (originPointNow - oringinPoint)/stepLenUse
        return jacVec



    # 计算当前时刻到辅助控制率
    def getAssistantControl(self, controlSeq, seqNumNow, seqSize, initialConditions,
                            initialControls, initialLSTMSeq, returnNum=0, modelType='NN'):
        xTemp = initialConditions[[0,1,3,4]] - self.xs
        dtvx1 = 2*self.P.dot(xTemp[[0,1]])   # 向量
        dtvx2 = 2*self.P.dot(xTemp[[2,3]])

        nowControls = np.array([0,0,0,0])            # 零输入
        nowConditions = np.array(initialConditions)               # 当前状态量
        nowInput = np.append(nowConditions, nowControls)
        outputTemp = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) # 零输入响应
        outputTemp = outputTemp[[0,1,3,4]] - self.xs

        ftilde1 = outputTemp[[0,1]]
        ftilde2 = outputTemp[[2,3]]
        LfV1 = dtvx1.dot(ftilde1)
        LfV2 = dtvx2.dot(ftilde2)

        # 第一个罐子的控制率      
        g1 = np.array([q/V, 0])
        Lg1V = dtvx1.dot(g1)
        if Lg1V == 0:
            h1_1xtk = 0
        else:
            h1_1xtk = 0
        g2 = np.array([0, 1/(rho*Cp*V)])
        Lg2V = dtvx1.dot(g2)
        if Lg2V == 0:
            h1_2xtk = 0
        elif Lg2V > 5e5:
            h1_2xtk = 5e5
        elif Lg2V < -5e5:
            h1_2xtk = -5e5
        else:
            h1_2xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg2V, 4))) / (Lg2V)
        
        g1 = np.array([q/V, 0])
        Lg1V = dtvx2.dot(g1)
        if Lg1V == 0:
            h2_1xtk = 0
        else:
            h2_1xtk = 0
        g2 = np.array([0, 1/(rho*Cp*V)])
        Lg2V = dtvx2.dot(g2)
        if Lg2V == 0:
            h2_2xtk = 0
        elif Lg2V > 5e5:
            h2_2xtk = 5e5
        elif Lg2V < -5e5:
            h2_2xtk = -5e5
        else:
            h2_2xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg2V, 4))) / (Lg2V)
        # 辅助控制率计算Lyapunov上限
        deltU = np.array([h1_1xtk, h1_2xtk, h2_1xtk, h2_2xtk])      
        return deltU
    
    def calConstraints(self, controlSeq, seqNumNow, seqSize,
                       initialConditions, initialControls,
                       returnNum, initialLSTMSeq, anotherContrlSeq,
                       modelType='NN'):
        # 此函数为first-principle版本
        # controlSeq: 目前为某个传感器的控制序列，2个一组
        # seqNumNow： 未来串口中的第几个
        # seqSize： MPC窗口大小
        # initialConditions：系统在MPC开始时候的系统状态
        # initialControls： 系统在MPC开始时候的控制量
        anotherContrlSeq = np.array(anotherContrlSeq)
        length = len(controlSeq)
        xTemp = np.array(initialConditions[[0,1,3,4]] - self.xs)
        x1Temp = xTemp[[0,1]]
        x2Temp = xTemp[[2,3]]
        if returnNum == 0:
            xkValue = x1Temp.T.dot(self.P).dot(x1Temp)
        elif returnNum == 1:
            xkValue = x2Temp.T.dot(self.P).dot(x2Temp)
        dtvx1 = 2*self.P.dot(xTemp[[0,1]])   # 向量
        dtvx2 = 2*self.P.dot(xTemp[[2,3]])
        dtvxTotal = 2*self.P2.dot(xTemp)
        
        # 先计算辅助控制率
        sontagControl = self.getAssistantControl(controlSeq, seqNumNow, seqSize, initialConditions,
                                                initialControls, initialLSTMSeq, returnNum, self.choice)
        if returnNum == 0:
            sontagUse = sontagControl[[0,1]]
            anotherContrlSeq = anotherContrlSeq.reshape(seqSize, int(length/seqSize))  
            # anotherContrl = sontagControl[[2,3]]
            # anotherContrlSeq = np.tile(anotherContrl, (seqSize,1))
        elif returnNum == 1:
            sontagUse = sontagControl[[2,3]]
            anotherContrlSeq = anotherContrlSeq.reshape(seqSize, int(length/seqSize))  
            # distributed 用下面这句话
            # anotherContrl = sontagControl[[0,1]]      # 使用另一个传过来的数值，如果是分布式的，这2句话需要有
            # anotherContrlSeq = np.tile(anotherContrl, (seqSize,1))
        else:
            print("wrong Number!!")
        # 辅助控制率和要优化的控制率拼接成一个控制序列
        controlSeq = np.array(controlSeq)
        controlSeq = controlSeq.reshape(seqSize, int(length/seqSize))
        if returnNum == 0:
            controlSeq = np.hstack((controlSeq, anotherContrlSeq))
        elif returnNum == 1:
            controlSeq = np.hstack((anotherContrlSeq, controlSeq))
        else:
            print("Wrong Number calConstraints")
        controlSeq = controlSeq.flatten()
              
        if (xkValue <= self.rhonn) and (seqNumNow > 0):
            for i in range(0,seqNumNow):
                deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
                nowControls = self.us + deltU           # 当前控制量
                nowConditions = np.array(initialConditions)               # 当前状态量
                nowInput = np.append(nowConditions, nowControls)
                newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
                initialConditions = newOutput
                initialControls = nowControls
                xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            V1 = xTemp[[0,1]].T.dot(self.P).dot(xTemp[[0,1]])
            V2 = xTemp[[2,3]].T.dot(self.P).dot(xTemp[[2,3]])
            if returnNum == 0:
                return -(V1 - self.rhonn)
                # return -(V1 - self.rhonn)
            elif returnNum == 1:
                return -(V2 - self.rhonn)
                # return -(V2 - self.rhonn)
            else: 
                print("wrong returnNumber!!!!!")
        else:
            deltU = sontagControl
            nowControls = self.us + deltU           # 当前控制量
            # nowControls = np.array(initialControls) + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
            xSontag = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            xSontag1 = xSontag[[0,1]]
            xSontag2 = xSontag[[2,3]]
            V1Sontag = dtvx1.dot(xSontag1)
            V2Sontag = dtvx2.dot(xSontag2)

            # 优化控制率计算Lyapunov约束项值
            if returnNum == 0:
                deltU = np.append(sontagUse, np.array([controlSeq[2], controlSeq[3]]))
            elif returnNum == 1:
                deltU = np.append(np.array([controlSeq[0], controlSeq[1]]), sontagUse)
            else:
                print("Wrong Number!!!!!!")
            nowControls = self.us + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
            optimalX = newOutput[[0,1,3,4]] - self.xs
            optimalX1 = optimalX[[0,1]]
            optimalX2 = optimalX[[2,3]]
            V1Optimal = dtvx1.dot(optimalX1)
            V2Optimal = dtvx2.dot(optimalX2)


            if returnNum == 0:
                return -(V1Optimal - V1Sontag)
            elif returnNum == 1:
                return -(V2Optimal - V2Sontag)
            else:
                print("wrong returnNumber!!!!!")
                
    def getConstraintJacVec(self, controlSeq, seqNumNow, seqSize,
                            initialConditions, initialControls,
                            returnNum, initialLSTMSeq, anotherContrlSeq, modelType='NN'):
        # seqNum       当前预测时间步
        # seqSize      MPC预测窗口大小
        # initialConditions 当前时刻的未开始MPC的状态
        # initialControls 当前时刻为开始MPC的控制量
        # controlNum    控制器编号 输入为0，1，2，。。。分别对应1，2，3.。。号控制器
        # modelType    预测模型类别
        anotherContrlSeq = np.array(anotherContrlSeq)
        length = len(controlSeq)
        stepLen = loopNum * sampleInterval
        controlSeq = np.array(controlSeq)
        controlSeqUsed = np.array(controlSeq)
        jacVec = np.zeros(controlSeq.shape)

        sontagControl = self.getAssistantControl(controlSeq, seqNumNow,
                                                seqSize, initialConditions,
                                                initialControls, initialLSTMSeq, 
                                                returnNum, self.choice)

        if returnNum == 0:
            sontagUse = sontagControl[[0,1]]
            # anotherContrl = sontagControl[[2,3]]
            anotherContrlSeq = anotherContrlSeq.reshape(seqSize, int(length/seqSize))
        elif returnNum == 1:
            sontagUse = sontagControl[[2,3]]
            # anotherContrl = sontagControl[[0,1]]
            anotherContrlSeq = anotherContrlSeq.reshape(seqSize, int(length/seqSize))
        else:
            print("wrong Number!!")
        # anotherContrlSeq = np.tile(anotherContrlSeq, (seqSize,1))
        controlSeq = np.array(controlSeq)
        controlSeq = controlSeq.reshape(seqSize, int(length/seqSize))
        if returnNum == 0:
            controlSeq = np.hstack((controlSeq, anotherContrlSeq))
        elif returnNum == 1:
            controlSeq = np.hstack((anotherContrlSeq, controlSeq))
        else:
            print("Wrong Number calConstraints")
        controlSeq = controlSeq.flatten()
        # anotherContrlSeq = anotherContrlSeq.flatten()

        # 更新当前状态至第 seqNum 个预测时间窗口
        for i in range(seqNumNow):
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
            nowControls = self.us+ deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
            initialConditions = newOutput
            initialLSTMSeq = self.updateInputSeq(initialLSTMSeq, nowInput)
        
        # 计算当前约束条件的基础值，用于求解约束条件的倒数
        controllerVarNum = int(len(controlSeq) / seqSize / 2)
        oringinPoint = self.calConstraints(controlSeqUsed, seqNumNow, seqSize,
                                        initialConditions, initialControls, 
                                        returnNum, initialLSTMSeq, anotherContrlSeq, 
                                        modelType)

        # 定位到第几个窗口         
        for j in range(controllerVarNum):
            if j % 2 ==0:
                stepLenUse = stepLen
            else:
                stepLenUse = stepLen * 1
            controlPosition = j
            controlSeqNew = np.zeros(controlSeqUsed.shape) + controlSeqUsed
            controlSeqNew[controlPosition] = controlSeqUsed[controlPosition] + stepLenUse
            originPointNow = self.calConstraints(controlSeqNew, seqNumNow, seqSize,
                                                initialConditions, initialControls, 
                                                returnNum, initialLSTMSeq, anotherContrlSeq,
                                                modelType)
            jacVec[j] = (originPointNow - oringinPoint)/stepLenUse
        return jacVec

        

    def distributedLMPC(self, initialConditions, initialControls, seqSize, initialLSTMSeq, controlNum, anotherControlSeq):
        # initialConditions: 初始状态
        # initialControls: 初始控制量
        # seqSize: MPC窗口长度
        # controlNum: 0:第一个控制器， 1:第二个控制器
        # return: CA, Q
        controlSeqInitial = np.tile(np.zeros(2), (1, seqSize))  # 2个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
                                            # 优化项的初始值
        controlSeqInitial = np.squeeze(controlSeqInitial)
        bound = []
        for i in range(seqSize):
            bound.append((-3.5, 3.5))
            bound.append((-5e5, 5e5))
        bound = tuple(bound)

        def calCons(controlSeqInitial):
            cons = []
            n = seqSize
            for i in range(n):
                argrow = (i, seqSize, initialConditions, initialControls, controlNum, initialLSTMSeq, anotherControlSeq, self.choice)
                cons.append({'type':'ineq', 'fun': self.calConstraints, 'jac': self.getConstraintJacVec, 'args':argrow})
            return cons
        
        if controlNum == 0:
            res = minimize(self.calObjective, controlSeqInitial,
                args=(seqSize, initialConditions, initialControls, initialLSTMSeq, controlNum, anotherControlSeq),
                bounds=bound, method='SLSQP', jac=self.getObjJac, constraints=calCons(controlSeqInitial))
        elif controlNum == 1:
            res = minimize(self.calObjective, controlSeqInitial,
                args=(seqSize, initialConditions, initialControls, initialLSTMSeq, controlNum, anotherControlSeq),
                bounds=bound, method='SLSQP', jac=self.getObjJac, constraints=calCons(controlSeqInitial))
        else:
            print("wrong calcons number!!")
        return res.x