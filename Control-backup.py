import numpy as np
import torch
import matplotlib.pyplot as plt
from PredictionModel import LSTM, device, LSTM2
# from pyomo.environ import * 
from Kinetics import cstr_cascading_kinetics
from scipy.optimize import minimize, NonlinearConstraint
import ipyopt

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

class controlMethod():
    def __init__(self):
        # 稳定点数值
        self.CA1s = 1.954
        self.CA2s = 1.954
        self.CA10s = 4
        self.CA20s = 4
        self.Q1s = 0
        self.Q2s = 0
        self.T1s = 401.9
        self.T2s = 401.9
        self.rhonn = 10
        self.rhomin = 12
        self.rho_1_2 = 760
        self.xs = np.array([self.CA1s, self.T1s, self.CA2s, self.T2s])   # 稳定点     ???????????????
        self.P = np.array([[1060, 22],[22, 0.52]])  # size: 4x4
        self.Q = np.diag([2e3, 1, 2e3, 1])      # size： 4x4
        self.R = np.diag([8e-13, 0.001])        # size：2x2
        self.PATH = "netModel/LSTM2_2_1_10epoch.pth"        ## 两个子系统的运行合并到一个LSTM中来做
        # self.PATH = "net/Model/LSTM_1_2_30epoch_1_windowSize.pth"   ## 每个子系统的运行是一个LSTM
        self.model =torch.load(self.PATH)
        self.choice = 'FirstPrinciple'    # or  'NN'   'FirstPrinciple'
        # self.choice = 'NN'
        self.fpModel = cstr_cascading_kinetics(Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)

        # for ipyopt used
        self.seqSize = 5
        self.initialConditions = np.zeros(1)
        self.initialControls = np.zeros(1)

    def nextStep(self, inputVar, method='NN'):
        if method == 'NN':
            output = self.model(inputVar)
            output = output.data.cpu().numpy()
        elif method == 'FirstPrinciple':
            initialConditions = inputVar[0:6]
            controlQuantity = inputVar[6:]
            output = self.fpModel.nextState(initialState=initialConditions, controlQuantity=controlQuantity, timeInterval=sampleInterval, loopNum=loopNum)
        return np.array(output)

    def calSontag():
        pass

    
    def calObjective(self, controlSeq, seqSize, initialConditions, initialControls):
        # 此函数为first-principle版本
        # controlSeq, 4个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
        sum = 0
        for i in range(0, seqSize):
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])

            nowControls = np.array(initialControls) + deltU           # 当前控制量
            u1Temp = nowControls[[0,1]]
            u2Temp = nowControls[[2,3]]
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, method=self.choice) 
            initialConditions = newOutput
            initialControls = nowControls
            xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            sum = sum + xTemp.T.dot(self.Q).dot(xTemp) + u1Temp.T.dot(self.R).dot(u1Temp) + \
                u2Temp.T.dot(self.R).dot(u2Temp)
            # sum = sum + xTemp.T*self.Q*xTemp + u1Temp.T*self.R*u1Temp + u2Temp.T*self.R*u2Temp
        return sum

    def calConstraints(self, controlSeq, seqNumNow, seqSize, initialConditions, initialControls, returnNum, modelType='NN'):
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
        # xkValue = x1Temp.T*self.P*x1Temp + x2Temp.T*self.P*x2Temp

        if (xkValue <= self.rhonn) or (seqNumNow > 0):       # 只需要判断第一个状态即可
            for i in range(0,seqNumNow):
                deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
                nowControls = np.array(initialControls) + deltU           # 当前控制量
                nowConditions = np.array(initialConditions)               # 当前状态量
                nowInput = np.append(nowConditions, nowControls)
                newOutput = self.nextStep(nowInput, method=self.choice) 
                initialConditions = newOutput
                initialControls = nowControls
                xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            # V = xTemp[0,1].T*self.P*xTemp[0,1] + xTemp[2,3].T*self.P*xTemp[2,3]
            V1 = xTemp[[0,1]].T.dot(self.P).dot(xTemp[[0,1]])
            V2 = xTemp[[2,3]].T.dot(self.P).dot(xTemp[[2,3]])
            # return (V1 <= self.rhonn) and (V2 <= self.rhonn)
            # return V1 - self.rhonn, V2 - self.rhonn, xTemp[[0,1]], xTemp[[2,3]]
            if returnNum == 0:
                return -(V1 - self.rhonn)
            elif returnNum == 1:
                return -(V2 - self.rhonn)
            else:
                print("wrong returnNumber!!!!!")

        # elif (xkValue <= self.rho_1_2) :
        else:
            xTemp = initialConditions[[0,1,3,4]] - self.xs      # x(k) 用于计算 dV(xk)/dx
            # for i in range(seqNumNow):
            #     deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
            #     nowControls = np.array(initialControls) + deltU           # 当前控制量
            #     nowConditions = np.array(initialConditions)               # 当前状态量
            #     nowInput = np.append(nowConditions, nowControls)
            #     newOutput = self.nextStep(nowInput, method=self.choice) 
            #     initialConditions = newOutput
            #     initialControls = nowControls
            #     xTemp = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            
            # sontag辅助控制率计算过程
            dtvx1 = 2*self.P.dot(xTemp[[0,1]])   # 向量
            dtvx2 = 2*self.P.dot(xTemp[[2,3]])

            nowControls = np.array([0,0,0,0])            # 零输入
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            outputTemp = self.nextStep(nowInput, method=self.choice) # 零输入响应
            outputTemp = outputTemp[[0,1,3,4]] - self.xs

            ftilde1 = outputTemp[[0,1]]
            ftilde2 = outputTemp[[2,3]]
            LfV1 = dtvx1.dot(ftilde1)
            LfV2 = dtvx2.dot(ftilde2)

            # 第一个罐子的控制量        # 这个地方用了动力学

            g1 = np.array([q/V, 0])
            Lg1V = dtvx1.dot(g1)
            if Lg1V == 0:
                h1_1xtk = 0
            else:
                h1_1xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg1V, 4)))/(Lg1V*Lg1V)
            g2 = np.array([0, 1/(rho*Cp*V)])
            Lg2V = dtvx1.dot(g2)
            if Lg2V == 0:
                h1_2xtk = 0
            else:
                h1_2xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg2V, 4))) / (Lg2V * Lg2V)
            
            # 第二个罐子的控制量
            g1 = np.array([q/V, 0])
            Lg1V = dtvx2.dot(g1)
            if Lg1V == 0:
                h2_1xtk = 0
            else:
                h2_1xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg1V, 4)))/ (Lg1V * Lg1V)
            g2 = np.array([0, 1/(rho*Cp*V)])
            Lg2V = dtvx2.dot(g2)
            if Lg2V == 0:
                h2_2xtk = 0
            else:
                h2_2xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg2V, 4))) / (Lg2V * Lg2V)

            # 辅助控制率计算Lyapunov上限
            deltU = np.array([h1_1xtk, h1_2xtk, h2_1xtk, h2_2xtk])      
            nowControls = np.array(initialControls) + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, method=self.choice) 
            xSontag = newOutput[[0,1,3,4]] - self.xs  # 只是取出来需要的两个状态
            xSontag1 = xSontag[[0,1]]
            xSontag2 = xSontag[[2,3]]
            V1Sontag = dtvx1.dot(xSontag1)
            V2Sontag = dtvx2.dot(xSontag2)

            # 优化控制率计算Lyapunov约束项值
            deltU = np.array([controlSeq[0], controlSeq[1], controlSeq[2], controlSeq[3]])
            nowControls = np.array(initialControls) + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, method=self.choice) 
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

    def getConstraintJacVec(self, controlSeq, seqNum, seqSize, initialConditions, initialControls, returnNum, modelType='NN'):
        #                   self, controlSeq, seqNumNow, seqSize, initialConditions, initialControls, returnNum
        # seqNum       当前预测时间步
        # seqSize      MPC预测窗口大小
        # initialConditions 当前时刻的未开始MPC的状态
        # initialControls 当前时刻为开始MPC的控制量
        # controlNum    控制器编号 输入为0，1，2，。。。分别对应1，2，3.。。号控制器
        # modelType    预测模型类别
        
        stepLen = 0.001     # 求微分时候的步长
        controlSeq = np.array(controlSeq)
        jacVec = np.zeros(controlSeq.shape)     # 约束导数的大小

        # 更新当前状态至第 seqNum 个预测时间窗口
        for i in range(seqNum):
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
            nowControls = np.array(initialControls) + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, method=self.choice) 
            initialControls = nowControls
            initialConditions = newOutput
        
        # 计算当前约束条件的基础值，用于求解约束条件的倒数
        controllerVarNum = int(len(controlSeq) / seqSize)
        oringinPoint = self.calConstraints(controlSeq, seqNum, seqSize, initialConditions, initialControls, returnNum, modelType)

        # 定位到第几个窗口                      # 这个地方不对！！！！！！！！！！？？？？？？？？？？
        for j in range(controllerVarNum):
            controlPosition = j + seqNum * 4
            controlSeqNew = controlSeq
            controlSeqNew[controlPosition] = controlSeqNew[controlPosition] + stepLen
            originPointNow = self.calConstraints(controlSeqNew, seqNum, seqSize, initialConditions, initialControls, returnNum, modelType)
            jacVec[j] = (originPointNow - oringinPoint)/stepLen

        return jacVec

    def getConstraintJacVec_new(self, controlSeq, seqNum, seqSize, initialConditions, initialControls, returnNum, selectNum, modelType='NN'):
        #                   self, controlSeq, seqNumNow, seqSize, initialConditions, initialControls, returnNum
        # seqNum       当前预测时间步
        # seqSize      MPC预测窗口大小
        # initialConditions 当前时刻的未开始MPC的状态
        # initialControls 当前时刻为开始MPC的控制量
        # controlNum    控制器编号 输入为0，1，2，。。。分别对应1，2，3.。。号控制器
        # selectNum     输出当前窗口第几维数据，例如，如果是第二个 窗口的第三维度数据，则seqNum=1, selectNum=2
        # modelType    预测模型类别
        
        stepLen = 0.001     # 求微分时候的步长
        controlSeq = np.array(controlSeq)
        jacVec = np.zeros(controlSeq.shape)     # 约束导数的大小

        # 更新当前状态至第 seqNum 个预测时间窗口
        for i in range(seqNum):
            deltU = np.array([controlSeq[4*i], controlSeq[4*i+1], controlSeq[4*i+2], controlSeq[4*i+3]])
            nowControls = np.array(initialControls) + deltU           # 当前控制量
            nowConditions = np.array(initialConditions)               # 当前状态量
            nowInput = np.append(nowConditions, nowControls)
            newOutput = self.nextStep(nowInput, method=self.choice) 
            initialControls = nowControls
            initialConditions = newOutput
        
        # 计算当前约束条件的基础值，用于求解约束条件的倒数
        controllerVarNum = int(len(controlSeq) / seqSize)
        oringinPoint = self.calConstraints(controlSeq, seqNum, seqSize, initialConditions, initialControls, returnNum, modelType)

        # 定位到第几个窗口
        for j in range(controllerVarNum):
            controlPosition = j + seqNum * 4
            controlSeqNew = controlSeq
            controlSeqNew[controlPosition] = controlSeqNew[controlPosition] + stepLen
            originPointNow = self.calConstraints(controlSeqNew, seqNum, seqSize, initialConditions, initialControls, returnNum, modelType)
            jacVec[j] = (originPointNow - oringinPoint)/stepLen

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
        return hessianMatrix
    
    def CentralizeLMPC(self, initialConditions, initialControls, seqSize):
        # initialConditions: 初始状态
        # initialControls: 初始控制量
        # seqSize: MPC窗口长度
        # return: CA, Q

        ###############################################################################################################
        ########## Pyomo方法  #############
        ###############################################################################################################
        # OPmodel = ConcreteModel()
        # OPmodel.deltCA = Var(range(0,2*seqSize), within=Reals, bounds=(-3.5,3.5))        # 入料量  优化变量
        # OPmodel.deltQ = Var(range(0,2*seqSize), within = Reals, bounds=(-5e5,5e5))       # 加热器  优化变量

        # OPmodel.f = Objective(rule= self.calObjective(OPmodel.deltCA, OPmodel.deltQ, seqSize, initialConditions, initialControls))
        # for nowTime in range(0,seqSize):
        #     OPmodel.c1 = Constraint(expr= self.calConstraints(seqNumNow = nowTime, controlSeqCA=OPmodel.deltCA, controlSeqQ = OPmodel.deltQ, seqSize = seqSize,\
        #         initialConditions=initialControls, initialControls=initialControls))
        
        # OPmodel.pprint()
        # SolverFactory('ipopt').solve(OPmodel).write()
        # print(OPmodel.deltCA)
        # print(OPmodel.deltQ)
        # # OPmodel.c1 = Constraint(OPmodel.nowTime, expr = self.calConstraints(seqNumNow = nowTime, controlSeqCA=OPmodel.deltCA, controlSeqQ = OPmodel.deltQ, seqSize = seqSize,\
        # #     initialConditions=initialControls, initialControls=initialControls) for nowTime in range(0,seqSize))        # using lambda function??????????????????????
        
        # return OPmodel.deltCA, OPmodel.deltQ
        ###############################################################################################################
        ###############################################################################################################

        ###############################################################################################################
        ######## Scipy.optimize  方法
        ###############################################################################################################
        # controlSeqInitial = np.tile(np.zeros(4), (1, seqSize))  # 4个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
        #                                     # 优化项的初始值
        # controlSeqInitial = np.squeeze(controlSeqInitial)
        # bound = []
        # for i in range(seqSize):        # 计算输入约束
        #     bound.append((-3.4, 3.4))
        #     bound.append((-5e5,5e5))
        #     bound.append((-3.4,3.4))
        #     bound.append((-5e5, 5e5))
        
        # bound = tuple(bound)

        # def calCons(controlSeqInitial):
        #     cons = []
        #     jacs = []
        #     # n = int(len(controlSeqInitial)/4)
        #     n = seqSize
        #     for i in range(n):        # 计算约束    
        #         argrow0 = (i, seqSize, initialConditions, initialControls,0, self.choice)
        #         argrow1 = (i, seqSize, initialConditions, initialControls,1, self.choice)     

        #         cons.append({'type':'ineq', 'fun': self.calConstraints, 'jac': self.getConstraintJacVec, 'args':argrow0})
        #         cons.append({'type':'ineq', 'fun': self.calConstraints, 'jac': self.getConstraintJacVec, 'args':argrow1})

        #     return cons

        # def func(x):
        #     return self.calObjective(x, seqSize, initialConditions, initialControls)
        
        # # res = minimize(self.calObjective, controlSeqInitial, args=(seqSize, initialConditions, initialControls), bounds=bound, method='trust-constr',constraints=calCons(controlSeqInitial), options={'disp':True})
        # # res = minimize(self.calObjective, controlSeqInitial, args=(seqSize, initialConditions, initialControls), bounds=bound, method='BFGS',constraints=calCons(controlSeqInitial), options={'disp':True})
        # res = minimize(self.calObjective, controlSeqInitial, args=(seqSize, initialConditions, initialControls), bounds=bound, method='SLSQP',constraints=calCons(controlSeqInitial), tol = 1e-10, options={'disp':True})
        # return res.x


        ###############################################################################################################
        ######## ipopt 方法
        ###############################################################################################################
        self.seqSize = seqSize
        self.initialConditions = initialConditions
        self.initialControls = initialControls

        controlSeqInitial = np.tile(np.array([0.0,0.0,0.0,0.0]), (1, seqSize))  # 4个一组，+0， +1， 为第一个控制器的 CA0 和 Q ， +2，+3，为第二个控制器的 CA0 和 Q
                                            # 优化项的初始值
        controlSeqInitial = np.squeeze(controlSeqInitial)

        x_l = np.tile(np.array([-3.5, -5e5, -3.5, -5e5]), (1,seqSize))
        x_l = np.squeeze(x_l)
        x_u = np.tile(np.array([3.5, 5e5, 3.5, 5e5]), (1,seqSize))
        x_u = np.squeeze(x_u)

        constraintsNum = seqSize * 2         # 约束的数量
        g_l = np.tile(np.array([0.0, 0.0]), (1, seqSize))
        g_l = np.squeeze(g_l)
        g_u = np.tile(np.array([1e15,1e15]), (1,seqSize))
        g_u = np.squeeze(g_u)

        sparsity_indices_jac_g = self.get_sparsity_g_jac_matrix(controlSeqInitial.shape[0])
        sparsity_indices_h = self.get_sparsit_hessian_matrix(controlSeqInitial.shape[0])

        # 不知道怎么求 Hessian 矩阵 , ipyopt 说可以不用求
        
        def func(x):
            return self.calObjective(x, seqSize, initialConditions, initialControls)
        def grad_f(x, out):
            varNum = seqSize * 4
            """Return the gradient of the objective"""
            for i in range(0,varNum,2):
                out[i] = 2*self.R[0,0] * x[i]
                out[i + 1] = 2 * self.R[1,1] * x[i + 1]
            return out
        
        def g(x, out):
            n = seqSize
            for i in range(n):
                out[2*i] = self.calConstraints(x, i, seqSize, initialConditions, initialControls, 0, self.choice)
                out[2*i + 1] = self.calConstraints(x, i, seqSize, initialConditions, initialControls, 1, self.choice)
            return out 
        
        def jac_g(x, out):
            varNum = seqSize * 4
            for i in range(seqSize):
                temp1 = self.getConstraintJacVec(x, i, seqSize, initialConditions, initialControls,0, self.choice)
                temp2 = self.getConstraintJacVec(x, i, seqSize, initialConditions, initialControls,1, self.choice)
                temp = np.append(temp1, temp2)
                for j in range(0, varNum):
                    out[varNum*i + j] = temp[j]
            return out

        # nlp = ipyopt.Problem(
        #     n = 4*seqSize,
        #     x_l = x_l,
        #     x_u = x_u,
        #     m = constraintsNum,
        #     g_l = g_l,
        #     g_u = g_u,
        #     sparsity_indices_jac_g = sparsity_indices_jac_g,
        #     sparsity_indices_h=sparsity_indices_h,
        #     func,              #   目标函数
        #     grad_f,
        #     g,
        #     jac_g,
        # )

        nlp = ipyopt.Problem(
            4*seqSize,
            x_l,
            x_u,
            constraintsNum,
            g_l,
            g_u,
            sparsity_indices_jac_g,
            sparsity_indices_h,
            func,
            grad_f,
            g,
            jac_g,
        )


        x, obj, status = nlp.solve(x0=controlSeqInitial)
        return x

        
        
        


# model = ConcreteModel() # 创建模型对象
# # define model variables
# # domain = Reals(Default) / NonNegativeReals Binary
# model.x1 = Var(domain=Reals)
# model.x2 = Var(domain=Reals)
# # define objective function
# # sense = minimize(Default) / maximize 
# model.f = Objective(expr = model.x1**2 + model.x2**2, sense=minimize)
# # define constraints, equations or inequations 
# model.c1 = Constraint(expr = -model.x1**2 + model.x2 <= 0)
# model.ceq1 = Constraint(expr = model.x1 + model.x2**2 == 2)
# # use 'pprint' to print the model information 
# model.pprint()
# SolverFactory('ipopt').solve(model).write()
# print(model.x1())
# print(model.x2())
# print(model.f())