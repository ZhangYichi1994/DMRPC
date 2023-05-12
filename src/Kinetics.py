from time import time
import matplotlib.pyplot as plt
import numpy as np 
# from ipywidgets import interact
# from IPython.display import display
from scipy.integrate import solve_ivp
import math
# from pyomo.environ import *

# CA1, T1, CB1, CA2, CA1, CA2, CB1, CB2, Q1, Q2, T1, T2 = y

class cstr_cascading_kinetics:
    def __init__(self, Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20):
        
        # parameters setting
        self.Ea  = Ea  #  72750     # activation energy J/gmol
        self.R   = R   #  8.314     # gas constant J/gmol/K
        self.k0  = k0  #  7.2e10    # Arrhenius rate constant 1/min
        self.V   = V   #  100.0     # Volume [L]
        self.rho = rho #  1000.0    # Density [g/L]
        self.Cp  = Cp  #  0.239     # Heat capacity [J/g/K]
        self.dHr = dHr #  -5.0e4    # Enthalpy of reaction [J/mol]
        self.q   = q   #  100.0     # Flowrate [L/min]

        self.Ti  = Ti  #  350.0     # Inlet feed temperature [K]
        self.cA0 = cA0 #  0.5      # Initial concentration [mol/L]
        self.T10 = T10 #  300.0    # Initial temperature of tank 1 [K]
        self.T20 = T20 #  300.0     # Initial temperature of tank 2 [K]
        # self.Q1  = Q1
        # self.Q2  = Q2
        # self.CA10= CA10
        # self.CA20= CA20 


    def k(self, T):
        return self.k0*np.exp(-self.Ea/self.R/T)
        # return self.k0*exp(-self.Ea/self.R/T)

    def systemDeriv(self, y, inputQuantity):
        # 系统状态
        # CA10， CA20， Q1，Q2 为输入
        # CA1，CA2， CB1，CB2， T1， T2为状态量
        
        CA1, T1, CB1, CA2, T2, CB2 = y
        CA10, Q1, CA20, Q2 = inputQuantity
        # first
        dCA1dt = (self.q/self.V)*(CA10 - CA1) - self.k(T1)*CA1*CA1
        dT1dt = (self.q/self.V)*(self.T10-T1) - (self.dHr/self.rho/self.Cp)*self.k(T1)*CA1*CA1 + (Q1/self.V/self.rho/self.Cp)
        dCB1dt = -(self.q/self.V)*CB1 + self.k(T1)*CA1*CA1
        # second 
        dCA2dt = (self.q/self.V)*(CA20+CA1) - ((2*self.q)/self.V)*CA2 - self.k(T2)*CA2*CA2
        dT2dt = (self.q/self.V) * (self.T20+T1) - ((2*self.q)/self.V)*T2 - (self.dHr/self.rho/self.Cp)*self.k(T2)*CA2*CA2 + (Q2/self.V/self.rho/self.Cp)
        dCB2dt = (self.q/self.V)*CB1 - ((2*self.q)/self.V)*CB2 + self.k(T2)*CA2*CA2
        return [dCA1dt, dT1dt, dCB1dt, dCA2dt, dT2dt, dCB2dt]

    def nextState(self, initialState, controlQuantity, timeInterval, loopNum):
        stateNow = initialState
        control = controlQuantity
        for i in range(0, loopNum):
            stateVar = self.systemDeriv(stateNow, control)
            stateVar = np.array(stateVar)
            stateNow = np.array(stateNow) + stateVar * timeInterval
            stateNow = stateNow.tolist()
        return stateNow

# ## Code Test Part
# # parameters setting
# Ea  = 50000     # activation energy J/gmol
# R   = 8.314     # gas constant J/gmol/K
# k0  = 8.46e6    # Arrhenius rate constant 1/min
# V   = 1         # Volume [L]
# rho = 1000.0    # Density [g/L]
# Cp  = 0.239     # Heat capacity [J/g/K]
# dHr = -1.15e4   # Enthalpy of reaction [J/mol]
# q   = 100.0     # Flowrate [L/min]
# cAi = 1.0       # Inlet feed concentration [mol/L]      ？？？？？？
# Ti  = 350.0     # Inlet feed temperature [K]
# cA0 = 0.5      # Initial concentration [mol/L]
# T10 = 300.0    # Initial temperature of tank 1 [K]
# T20 = 300.0     # Initial temperature of tank 2 [K]
# Q1 = 1e3
# Q2 = 1e3
# CA10 = 1.0
# CA20 = 1.0

# kinetic1 = cstr_cascading_kinetics( Ea, R, k0, V, rho, Cp, dHr, q, Ti, cA0, T10, T20)


# CA10 = cAi
# CA20 = cAi
# CA1 = cA0
# CA2 = cA0
# CB1 = cA0
# CB2 = cA0
# Q1 = 1e3
# Q2 = 1e3
# T1 = T10
# T2 = T20
# # simulation
# initialConditions = [CA1, T1, CB1, CA2, T2, CB2]
# sampleInterval = 0.0001 # 采样时间
# loopNum = 100           # 每回合采样次数

# controlQuantity = [CA10, Q1, CA20, Q2]
# state = kinetic1.nextState(initialState=initialConditions, controlQuantity=controlQuantity, timeInterval=sampleInterval, loopNum=loopNum)
# print(state)
# print("next State is:")

# controlQuantity = [CA10+0.5, Q1, CA20, Q2]
# initialConditions = state
# state = kinetic1.nextState(initialState=initialConditions, controlQuantity=controlQuantity, timeInterval=sampleInterval, loopNum=loopNum)

# print(state)