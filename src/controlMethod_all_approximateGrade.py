# 集中式LMPC
# 机理模型作为递推模型
# 使用近似方法来计算：Sontag辅助控制率、优化问题约束雅各比矩阵

from secrets import choice
import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel import LSTM, device, LSTM2, ANN
# from pyomo.environ import * 
from Kinetics import cstr_cascading_kinetics
from scipy.optimize import minimize, NonlinearConstraint
torch.set_default_tensor_type(torch.DoubleTensor)
# import ipyopt

# 如果使用神经网络，归一化没有考虑 ？？？？？？？？？？
# 

# 控制量相关内容
# 目标函数： L = xT*Q*x + u1T*R*u1 + u2T*R*u2,   Q= diag[2 × 103 1 2 × 103 1], R1 = R2 = diag[8e−13 0.001];
#   x要改成输入 deltU 的累积量
# 李雅普诺夫函数: V = x1T*P1*x1 + x2T*P2*x2,    P1=P2=np.array([[1060, 22], [22,0.52]])
# 约束条件： x in X, 
#           deltU in DELTU, 
#           dVdt * Fnn(x(tk), u1, u2) <= dVdt * Fnn(x(tk), Phi_nn1(tk), Phi_nn2(tk))    #在域外       
#            dVdt * Fnn(x(tk), u1, u2) <= rho_nn            # 在域内

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

class controlMethodApproximateGrade():      # 求解桑塔格控制率的时候，使用了机理模型中获得的导数
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
        self.rhonn = 50.0
        self.rhomin = 12.0
        self.rho_1_2 = 760.0
        self.xs = np.array([self.CA1s, self.T1s, self.CA2s, self.T2s])   # 稳定点     ???????????????
        self.us = np.array([self.CA10s, self.Q1s, self.CA20s, self.Q2s])
        self.P = np.array([[1060.0, 22.0],[22.0, 0.52]])  # size: 4x4
        self.P2 = np.array([[1060.0, 22.0, 0.0, 0.0],     # size: 8x8
                            [22.0, 0.52, 0.0, 0.0],
                            [0.0, 0.0, 1060.0, 22.0],
                            [0.0, 0.0, 22.0, 0.52]])
        self.Q = np.diag([2e3, 1.0, 2e3, 1.0])      # size： 4x4
        # self.R = np.diag([8e-13, 0.001])        # size：2x2
        self.R = np.diag([0.001, 8e-13])
        # self.PATH = "netModel/LSTM2_2_1_10epoch.pth"        ## 两个子系统的运行合并到一个LSTM中来做
        # self.PATH = "net/Model/LSTM_1_2_30epoch_1_windowSize.pth"   ## 每个子系统的运行是一个LSTM
        # self.PATH = "netModel/LSTM2_variation.pth"
        # self.model = LSTM2(6, 64, 6, 1, 11, 1).to(device)
        # self.model.load_state_dict = torch.load(self.PATH)['model']

        # self.PATH = "netModel/ANNvariation.pth"
        # self.model = ANN(10,64,32,6).to(device)
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
    def updateInputSeqWithoutNorm(self, oldInput, newInput):
        # 更新LSTM的输入序列
        # 不要归一化
        # 输入 输出 均为 numpy类型
        newInput = np.array(newInput)
        newSeq = np.zeros(oldInput.shape)
        newSeq[0:-1,:] = oldInput[1:,:]
        newSeq[-1,:] = newInput
        return newSeq

    def rollBack(self, lstmSeq, windowSize):
        length = lstmSeq.shape[0]
        if windowSize < (length - 5):             # 这个 5 与lstm的时间窗口长度有关
            controlSeq = lstmSeq[-(windowSize):,6:]
            newLSTMSeq = lstmSeq[-(windowSize + 5):-windowSize,:]           # 这个 5 为 LSTM所需要的序列长度
            for i in range(windowSize):
                newLSTMSeq = torch.Tensor(newLSTMSeq).to(device)
                newLSTMSeq = newLSTMSeq.unsqueeze(0)
                output = self.model(newLSTMSeq)
                output = output.data.cpu().numpy()
                output = np.squeeze(output)

                output = output * (self.maxOutputValue - self.minOutputValue + eps) + self.minOutputValue
                controlUse = controlSeq[i,:]
                controlUse = controlUse * (self.maxInputValue[6:] - self.minInputValue[6:] + eps) + self.minInputValue[6:]
                update = np.append(output, controlUse)

                newLSTMSeq = np.squeeze(newLSTMSeq)
                newLSTMSeq = newLSTMSeq.data.cpu().numpy()
                newLSTMSeq = self.updateInputSeq(newLSTMSeq, update)
            lstmHistory = lstmSeq
            lstmHistory[-5:,:] = newLSTMSeq

            newLSTMSeq = torch.Tensor(newLSTMSeq).to(device)
            newLSTMSeq = newLSTMSeq.unsqueeze(0)
            output = self.model(newLSTMSeq)
            output = output.data.cpu().numpy()
            output = np.squeeze(output)
            output = output * (self.maxOutputValue - self.minOutputValue + eps) + self.minOutputValue
            newLSTMSeq = np.squeeze(newLSTMSeq)
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
                    initialControls, initialLSTMSeq):
        # controlSeq, 4个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
        sum = 0
        for i in range(0, seqSize):
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])

            nowControls = self.us + deltU           # 当前控制量
            u1Temp = deltU[[0,1]]
            u2Temp = deltU[[2,3]]
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) ###########################################   改到了这里
            initialConditions = newOutput
            initialLSTMSeq = self.updateInputSeq(initialLSTMSeq, nowInput)
            xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            sum = sum + xTemp.T.dot(self.Q).dot(xTemp) + u1Temp.T.dot(self.R).dot(u1Temp) + \
                u2Temp.T.dot(self.R).dot(u2Temp)
            # sum = sum + xTemp.T.dot(self.Q).dot(xTemp)
            # sum = sum + xTemp.T*self.Q*xTemp + u1Temp.T*self.R*u1Temp + u2Temp.T*self.R*u2Temp
        return sum

    def getObjJac(self, controlSeq, seqSize, initialConditions, initialControls, initialLSTMSeq):
        initialConditions = np.array(initialConditions)
        controlSeq = np.array(controlSeq)
        sum = 0
        stepLen = sampleInterval * loopNum      # 求微分时候的步长、
        jacVec = np.zeros(controlSeq.shape)     # 约束导数的大小
        seqLen = controlSeq.shape[0]

        oringinPoint = self.calObjective(controlSeq, seqSize, initialConditions, initialControls, initialLSTMSeq)
        # 定位到第几个窗口                      
        for j in range(seqLen):
            if j % 2 == 0:
                stepLenUse = stepLen
            else:
                stepLenUse = stepLen * 1
            controlPosition = j
            controlSeqNew = np.zeros(controlSeq.shape) + controlSeq
            controlSeqNew[controlPosition] = controlSeq[controlPosition] + stepLenUse
            originPointNow = self.calObjective(controlSeqNew, seqSize, initialConditions, initialControls, initialLSTMSeq)
            jacVec[j] = (originPointNow - oringinPoint)/stepLenUse
        return jacVec

    def calConstraints(self, controlSeq, seqNumNow, seqSize, initialConditions,
                    initialControls, returnNum, initialLSTMSeq, modelType='NN'):
        # 此函数为first-principle版本
        # seqNumNow： 未来串口中的第几个
        # controlSeqCA： 第一个控制量，CA，控制序列
        # controlSeqQ：第二个控制量，Q， 控制序列
        # seqSize： MPC窗口大小
        # initialConditions：系统在MPC开始时候的系统状态
        # initialControls： 系统在MPC开始时候的控制量
        xTemp = np.array(initialConditions[[0,1,3,4]] - self.xs)
        x1Temp = xTemp[[0,1]]
        x2Temp = xTemp[[2,3]]

        xkValue = x1Temp.T.dot(self.P).dot(x1Temp) + x2Temp.T.dot(self.P).dot(x2Temp)
        # xkValue = xkValue - 1e-5
        # xkValue = x1Temp.T*self.P*x1Temp + x2Temp.T*self.P*x2Temp

        if (xkValue <= self.rhonn) or (seqNumNow > 0):       # 只需要判断第一个状态即可
            for i in range(0,seqNumNow):
                deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
                nowControls = self.us + deltU           # 当前控制量
                nowConditions = np.array(initialConditions)               # 当前状态量
                nowInput = np.append(nowConditions, nowControls)
                newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
                initialConditions = newOutput
                initialLSTMSeq = self.updateInputSeq(initialLSTMSeq, nowInput)
                xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            # V = xTemp[0,1].T*self.P*xTemp[0,1] + xTemp[2,3].T*self.P*xTemp[2,3]
            V1 = xTemp[[0,1]].T.dot(self.P).dot(xTemp[[0,1]])
            V2 = xTemp[[2,3]].T.dot(self.P).dot(xTemp[[2,3]])
            # return (V1 <= self.rhonn) and (V2 <= self.rhonn)
            # return V1 - self.rhonn, V2 - self.rhonn, xTemp[[0,1]], xTemp[[2,3]]
            if returnNum == 0:
                return -(V1 + V2 - self.rhonn)
                # return -(V1 - self.rhonn)
            elif returnNum == 1:
                return -(V1 + V2 - self.rhonn)
                # return -(V2 - self.rhonn)
            else: 
                print("wrong returnNumber!!!!!")

        # elif (xkValue <= self.rho_1_2) :
        else:
            xTemp = initialConditions[[0,1,3,4]] - self.xs      # x(k) 用于计算 dV(xk)/dx
            
            # sontag辅助控制率计算过程
            dtvx1 = 2*self.P.dot(xTemp[[0,1]])   # 向量
            dtvx2 = 2*self.P.dot(xTemp[[2,3]])
            dtvxTotal = 2*self.P2.dot(xTemp)

            nowControls = np.array([0,0,0,0])            # 零输入
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            outputTemp = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) # 零输入响应
            outputTemp = outputTemp[[0,1,3,4]] - self.xs

            ftilde1 = outputTemp[[0,1]]
            ftilde2 = outputTemp[[2,3]]
            LfV1 = dtvx1.dot(ftilde1)
            LfV2 = dtvx2.dot(ftilde2)
            LfVTotal = dtvxTotal.dot(outputTemp)

            # 第一个罐子的控制量        # 这个地方用了动力学    ###########
            stepLen = loopNum * sampleInterval     # 求微分时候的步长
            stepLen2 = stepLen * 1
            controlSeq = np.array(controlSeq)
            jacVec = np.zeros(controlSeq.shape)     # 约束导数的大小
            controlVarNum = int(len(controlSeq)/seqSize)
            # g = np.zeros([controlVarNum, int(controlVarNum/2)])
            g = np.zeros([controlVarNum, controlVarNum])
            for i in range(controlVarNum):

                if i % 2 == 0:
                    stepLenUse = stepLen
                else:
                    stepLenUse = stepLen2
                # deltU = np.array([controlSeq[0], controlSeq[1], controlSeq[2], controlSeq[3]])
                deltU = np.array([0.0, 0.0, 0.0, 0.0])
                nowControls = self.us + deltU
                nowConditions = np.array(initialConditions)
                nowInput = np.append(nowConditions, nowControls)
                # newOutput = self.nextStep(nowInput, self.choice)
                # newOutput = np.array(self.fpModel.systemDeriv(nowConditions, nowControls))
                newOutput = np.array(self.getOneStepGrad(nowConditions, nowControls, initialLSTMSeq, self.choice))
                xTemp = newOutput[[0,1,3,4]] - self.xs
                originalPoint1 = xTemp[[0,1]]
                originalPoint2 = xTemp[[2,3]]
                originalPointTotal = xTemp
                
                deltU_new = deltU
                deltU_new[i] = deltU[i] + stepLenUse 
                nowControls = self.us + deltU_new
                nowConditions = np.array(initialConditions)
                nowInput = np.append(nowConditions, nowControls)
                # newOutput = self.nextStep(nowInput, self.choice)
                # newOutput = np.array(self.fpModel.systemDeriv(nowConditions, nowControls))
                newOutput = np.array(self.getOneStepGrad(nowConditions, nowControls, initialLSTMSeq, self.choice))
                xTemp = newOutput[[0,1,3,4]] - self.xs
                originalPointNew1 = xTemp[[0,1]]
                originalPointNew2 = xTemp[[2,3]]
                originalPointNewTotal = xTemp
                
                # g[i,:] = (originalPointNew1 - originalPoint1) / stepLen
                # g[i+1, :] = (originalPointNew2 - originalPoint2) / stepLen
                g[i,:] = (originalPointNewTotal - originalPointTotal) / stepLenUse
                # g[2*i+2, :] = (originalPointNewTotal - originalPointTotal) / stepLen


            # 以上需要使用近似方法来近似g1
            g1 = g[0,:]
            # Lg1V = dtvx1.dot(g1)
            Lg1V = dtvxTotal.dot(g1)
            if Lg1V == 0:
                h1_1xtk = 0
            else:
                h1_1xtk = 0
                # h1_1xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg1V, 4)))/(Lg1V)
                # h1_1xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg1V, 4)))/(Lg1V*Lg1V)
            g2 = g[1,:]
            # Lg2V = dtvx1.dot(g2)
            Lg2V = dtvxTotal.dot(g2)
            if Lg2V == 0:
                h1_2xtk = 0
            elif Lg2V > 5e5:
                h1_2xtk = 5e5
            elif Lg2V < -5e5:
                h1_2xtk = -5e5
            else:
                h1_2xtk = -(LfVTotal + np.sqrt(LfVTotal*LfVTotal + np.power(Lg2V, 4))) / (Lg2V)
                # h1_2xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg2V, 4))) / (Lg2V)
                # h1_2xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg2V, 4))) / (Lg2V * Lg2V)
            
            # 第二个罐子的控制量
            # g1 = np.array([q/V, 0])
            g1 = g[2,:]
            # Lg1V = dtvx2.dot(g1)
            Lg1V = dtvxTotal.dot(g1)
            if Lg1V == 0:
                h2_1xtk = 0
            else:
                h2_1xtk = 0
                # h2_1xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg1V, 4)))/ (Lg1V)
                # h2_1xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg1V, 4)))/ (Lg1V * Lg1V)
            # g2 = np.array([0, 1/(rho*Cp*V)])
            g2 = g[3,:]
            # Lg2V = dtvx2.dot(g2)
            Lg2V = dtvxTotal.dot(g2)
            if Lg2V == 0:
                h2_2xtk = 0
            elif Lg2V > 5e5:
                h2_2xtk = 5e5
            elif Lg2V < -5e5:
                h2_2xtk = -5e5
            else:
                h2_2xtk = -(LfVTotal + np.sqrt(LfVTotal*LfVTotal + np.power(Lg2V, 4))) / (Lg2V)
                # h2_2xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg2V, 4))) / (Lg2V)
                # h2_2xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg2V, 4))) / (Lg2V * Lg2V)

            # 辅助控制率计算Lyapunov上限
            deltU = np.array([h1_1xtk, h1_2xtk, h2_1xtk, h2_2xtk])      
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
            deltU = np.array([controlSeq[0], controlSeq[1], controlSeq[2], controlSeq[3]])
            nowControls = self.us + deltU           # 当前控制量
            # nowControls = np.array(initialControls) + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
            optimalX = newOutput[[0,1,3,4]] - self.xs
            optimalX1 = optimalX[[0,1]]
            optimalX2 = optimalX[[2,3]]
            V1Optimal = dtvx1.dot(optimalX1)
            V2Optimal = dtvx2.dot(optimalX2)

            # return (V1Optimal <= V1Sontag) and (V2Optimal <= V2Sontag)
            # return V1Optimal - V1Sontag, V2Optimal - V2Sontag, xTemp[[0,1]], xTemp[[2,3]]
            if returnNum == 0:
                return -(V1Optimal - V1Sontag)
            elif returnNum == 1:
                return -(V2Optimal - V2Sontag)
            else:
                print("wrong returnNumber!!!!!")

    def getConstraintJacVec(self, controlSeq, seqNum, seqSize, initialConditions,
                            initialControls, returnNum, initialLSTMSeq, modelType='NN'):
        #                   self, controlSeq, seqNumNow, seqSize, initialConditions, initialControls, returnNum
        # seqNum       当前预测时间步
        # seqSize      MPC预测窗口大小
        # initialConditions 当前时刻的未开始MPC的状态
        # initialControls 当前时刻为开始MPC的控制量
        # controlNum    控制器编号 输入为0，1，2，。。。分别对应1，2，3.。。号控制器
        # modelType    预测模型类别
        
        stepLen = loopNum * sampleInterval     # 求微分时候的步长
        controlSeq = np.array(controlSeq)
        jacVec = np.zeros(controlSeq.shape)     # 约束导数的大小

        # 更新当前状态至第 seqNum 个预测时间窗口
        for i in range(seqNum):
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
            nowControls = self.us+ deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, initialLSTMSeq, method=self.choice) 
            initialConditions = newOutput
            initialLSTMSeq = self.updateInputSeq(initialLSTMSeq, nowInput)
        
        # 计算当前约束条件的基础值，用于求解约束条件的倒数
        controllerVarNum = int(len(controlSeq) / seqSize)
        oringinPoint = self.calConstraints(controlSeq, seqNum, seqSize, initialConditions, initialControls, returnNum, initialLSTMSeq, modelType)

        # 定位到第几个窗口         
        for j in range(controllerVarNum):
            if j % 2 ==0:
                stepLenUse = stepLen
            else:
                stepLenUse = stepLen * 1
            controlPosition = j + seqNum * 4
            controlSeqNew = np.zeros(controlSeq.shape) + controlSeq
            controlSeqNew[controlPosition] = controlSeq[controlPosition] + stepLenUse
            originPointNow = self.calConstraints(controlSeqNew, seqNum, seqSize, initialConditions, initialControls, returnNum, initialLSTMSeq, modelType)
            jacVec[j] = (originPointNow - oringinPoint)/stepLenUse
        return jacVec

    def get_sparsity_g_jac_matrix(self, varNum):
        # varNum = x.shape()[0]
        constraintNum = self.seqSize * 2        # 雅各比矩阵的长度 = MPC序列长度 * 变量个数
        jacMatrix = np.ones([constraintNum, varNum])
        row = []
        col = []
        # for i in range(0,constraintNum, 2):
        #     jacMatrix[i, 2*i] = 1
        #     jacMatrix[i, 2*i +1] = 1
        #     jacMatrix[i+1, 2*i +2] = 1
        #     jacMatrix[i+1, 2*i +3] = 1
        
        
        # for i in range(0, constraintNum):
        #     for j in range(0, varNum):
        #         if jacMatrix[i,j] == 1:
        #             row.append(i)
        #             col.append(j)
        
        for i in range(constraintNum):             # 列
            for j in range(varNum):   # 行
                if jacMatrix[i,j] == 1:
                    row.append(i)
                    col.append(j)
        row = np.array(row)
        col = np.array(col)
        return (row, col)

    def get_sparsit_hessian_matrix(self, varNum):
        hessianMatrix = np.zeros([varNum, varNum])
        for i in range(0,varNum,2):
            hessianMatrix[i,i] = 1
            hessianMatrix[i,i+1] = 1 
            hessianMatrix[i+1,i] = 1
            hessianMatrix[i+1, i+1] = 1
        # hessianMatrix = np.diag([1 for  i in range(varNum)])
        row = []
        col = []
        for i in range(varNum):
            for j in range(varNum):
                if hessianMatrix[i,j]== 1:
                    row.append(i)
                    col.append(j)
        row = np.array(row)
        col = np.array(col)

        return (row, col)
    
    def CentralizeLMPC(self, initialConditions, initialControls, seqSize, initialLSTMSeq):
        # initialConditions: 初始状态
        # initialControls: 初始控制量
        # seqSize: MPC窗口长度
        # return: CA, Q

        ###############################################################################################################
        ######## Scipy.optimize  方法
        ###############################################################################################################
        controlSeqInitial = np.tile(np.zeros(4), (1, seqSize))  # 4个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
                                            # 优化项的初始值
        controlSeqInitial = np.squeeze(controlSeqInitial)
        bound = []
        for i in range(seqSize):        # 计算输入约束
            bound.append((-3.5, 3.5))
            bound.append((-5e5,5e5))
            bound.append((-3.5,3.5))
            bound.append((-5e5, 5e5))
        
        bound = tuple(bound)

        def calCons(controlSeqInitial):
            cons = []
            jacs = []
            # n = int(len(controlSeqInitial)/4)
            n = seqSize
            for i in range(n):        # 计算约束    
                argrow0 = (i, seqSize, initialConditions, initialControls,0, initialLSTMSeq, self.choice)
                argrow1 = (i, seqSize, initialConditions, initialControls,1, initialLSTMSeq,self.choice)     

                cons.append({'type':'ineq', 'fun': self.calConstraints, 'jac': self.getConstraintJacVec, 'args':argrow0})
                cons.append({'type':'ineq', 'fun': self.calConstraints, 'jac': self.getConstraintJacVec, 'args':argrow1})
                # cons.append({'type':'ineq', 'fun': self.calConstraints, 'args':argrow0})
                # cons.append({'type':'ineq', 'fun': self.calConstraints, 'args':argrow1})

            return cons
        
        res = minimize(self.calObjective, controlSeqInitial,
                    args=(seqSize, initialConditions, initialControls, initialLSTMSeq),
                    bounds=bound, method='SLSQP', jac=self.getObjJac,
                    constraints=calCons(controlSeqInitial), tol = 1e-10, options={'disp':True})
        # res = minimize(self.calObjective, controlSeqInitial, args=(seqSize, initialConditions, initialControls), bounds=bound, method='SLSQP', constraints=calCons(controlSeqInitial), tol = 1e-10, options={'disp':True})

        return res.x   

    

    