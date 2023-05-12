# --utf-8--
# Data generation part
# The generated data is used for LSTM training


from matplotlib.pyplot import axis
import numpy as np

from Kinetics import *
from PredictionModel import *
from Kinetics import cstr_cascading_kinetics

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
sampleInterval = 0.0001     # 采样时间 - 微分时间
loopNum = 100               # 采样周期

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
T1s = 401.9
Q1s = 0.0

# generate  random input
# uTemp = np.random.rand(10000, 4)  # delt_A10, delt_Q1, delt_A20, delt_Q2
# u = uTemp * np.array([7, 1e6, 7, 1e6])
# u = u + np.array([0.5, -5e5, 0.5, -5e5])
# us = np.array([4.0, 0.0, 4.0, 0.0])   # 稳定点 

# generate obervation data with different input
# initial input
CA10 = cAi
CA20 = cAi
Q1 = 1e3
Q2 = 1e3
kinetic1 = cstr_cascading_kinetics( Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)
initialConditions = [CA1, T1, CB1, CA2, T2, CB2]

class generateDataWindow():
    def __init__(self, sampleSize, windowSize, feature, outputFeature,
                storePath, sampleInterval, loopNum):
        self.xs = np.array([CA1s, T1s, CA1s, T1s])
        self.us = np.array([CA10s, Q1s, CA10s, Q1s])
        self.P = np.array([[1060.0, 22.0],[22.0, 0.52]])
        self.fpmodel = cstr_cascading_kinetics(Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)
        self.sampleSize = sampleSize
        self.windowSize = windowSize
        self.feature = feature
        self.outputFeature = outputFeature
        self.pathStore = storePath
        self.sampleInterval = sampleInterval
        self.loopNum = loopNum
        # 用于存储数据
        self.inputRecord = np.zeros([self.sampleSize, self.windowSize, self.feature])
        self.outputRecord_derive = np.zeros([self.sampleSize, self.windowSize, self.outputFeature])
        self.outputRecord_state = np.zeros([self.sampleSize, self.windowSize, self.outputFeature])

    def generation(self):

        #### 如何保证生成的点全部在闭环状态里面？？？
        np.random.seed(6)
        t = 0
        while( t< self.sampleSize):
            if t % 100 == 0:
                print("Loop Time: ", t)
            initialConditions = np.random.rand(6) * np.array([4, 200, 4, 4, 200, 4]) + np.array([0.0, 300.0, 0.0, 0.0, 300.0, 0.0])
            # 用于 “临时” 存储数据 
            inputTemp = np.zeros([self.windowSize, self.feature])
            outputTemp_var = np.zeros([self.windowSize, self.outputFeature])
            outputTemp_state = np.zeros([self.windowSize, self.outputFeature])

            for i in range(self.windowSize):
                tempState = np.zeros(self.feature)
                tempState[0] = initialConditions[0]  # 当前状态
                tempState[1] = initialConditions[1]
                tempState[2] = initialConditions[2]
                tempState[3] = initialConditions[3] 
                tempState[4] = initialConditions[4]
                tempState[5] = initialConditions[5]  

                u = np.random.rand(4) * np.array([7.0, 10e5, 7.0, 10e5]) + \
                    np.array([0.5, -5e5, 0.5, -5e5]) # 
                # u = self.getSontagControl(initialConditions)    
                tempState[6] = u[0]   # 当前输入，第一级 入料
                tempState[7] = u[1]   # 当前输入，第一级 加热
                tempState[8] = u[2]   # 当前输入，第二级 入料
                tempState[9] = u[3]   # 当前输入，第二级 加热

                controlQuantity = [u[0], u[1], u[2], u[3]]
                variation = kinetic1.systemDeriv(initialConditions, controlQuantity)
                initialConditions = kinetic1.nextState(initialState=initialConditions, controlQuantity=controlQuantity, 
                                            timeInterval=self.sampleInterval, loopNum=self.loopNum)

                inputTemp[i, :] = np.array(tempState)
                outputTemp_var[i,:] = np.array(variation)
                outputTemp_state[i,:] = np.array(initialConditions)

            if (np.isnan(inputTemp).any() == False) and \
                (np.isnan(outputTemp_var).any() == False) and \
                (np.isnan(outputTemp_state).any() == False) and\
                (np.isinf(inputTemp).any() == False) and \
                (np.isinf(outputTemp_var).any() == False) and \
                (np.isinf(outputTemp_state).any() == False):
                # 状态记录
                self.inputRecord[t,:,:] = inputTemp
                self.outputRecord_derive[t,:,:] = outputTemp_var
                self.outputRecord_state[t,:,:] = outputTemp_state
                t = t + 1
            else:
                print("Relocated triggered!")   

        # np.save(self.pathStore + "inputRecord.npy", self.inputRecord)
        # np.save(self.pathStore + "outputRecord_variation.npy", self.outputRecord_derive)
        # np.save(self.pathStore + "outputRecord_state.npy", self.outputRecord_state)
        np.save(self.pathStore + "StateTraininputRecord_rand.npy", self.inputRecord)
        np.save(self.pathStore + "StateTrainoutputRecord_variation_rand.npy", self.outputRecord_derive)
        np.save(self.pathStore + "StateTrainoutputRecord_state_rand.npy", self.outputRecord_state)

        print("data generation success!")

        return self.inputRecord, self.outputRecord_derive, self.outputRecord_state


    def getSontagControl(self, initialConditions):
        initialConditions = np.array(initialConditions)
        xTemp = initialConditions[[0,1,3,4]] - self.xs
        # sontag辅助控制率计算过程
        dtvx1 = 2*self.P.dot(xTemp[[0,1]])   # 向量
        dtvx2 = 2*self.P.dot(xTemp[[2,3]])
        nowControls = np.array([0,0,0,0])            # 零输入
        nowConditions = np.array(initialConditions)               # 当前状态量
        outputTemp = self.fpmodel.nextState(nowConditions, nowControls,
                                        self.sampleInterval, self.loopNum) # 零输入响应
        outputTemp = np.array(outputTemp)
        outputTemp = outputTemp[[0,1,3,4]] - self.xs

        ftilde1 = outputTemp[[0,1]]
        ftilde2 = outputTemp[[2,3]]
        LfV1 = dtvx1.dot(ftilde1)
        LfV2 = dtvx2.dot(ftilde2)


        g1 = np.array([q/V, 0])
        Lg1V = dtvx1.dot(g1)
        if Lg1V == 0:
            h1_1xtk = 0
        else:
            h1_1xtk = 0
            # h1_1xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg1V, 4)))/(Lg1V)
            # h1_1xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg1V, 4)))/(Lg1V*Lg1V)
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
            # h1_2xtk = -(LfV1 + np.sqrt(LfV1*LfV1 + np.power(Lg2V, 4))) / (Lg2V * Lg2V)
        
        # 第二个罐子的控制量
        g1 = np.array([q/V, 0])
        Lg1V = dtvx2.dot(g1)
        if Lg1V == 0:
            h2_1xtk = 0
        else:
            h2_1xtk = 0
            # h2_1xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg1V, 4)))/ (Lg1V)
            # h2_1xtk = -(LfV2 + np.sqrt(LfV2*LfV2 + np.power(Lg1V, 4)))/ (Lg1V * Lg1V)
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

        deltU = np.array([h1_1xtk, h1_2xtk, h2_1xtk, h2_2xtk])
        deltTemp = np.random.rand(deltU.shape[0])
        deltU = deltU + deltTemp * 0.1
        u = deltU + self.us
        return u




        

sampleSize = 256000
windowSize = 5
feature = 10
outputFeature = 6
storePath = "data/windows/sontag/"
sampleInterval = 0.0001
loopNum = 100

a = generateDataWindow(sampleSize, windowSize, feature, outputFeature,
                storePath, sampleInterval, loopNum)
b, c, d = a.generation()
