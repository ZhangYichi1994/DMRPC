import matplotlib.pyplot as plt
import numpy as np
# from ipywidgets import interact
# from IPython.display import display
import math
from scipy.integrate import solve_ivp

# CA1, T1, CB1, CA2, CA1, CA2, CB1, CB2, Q1, Q2, T1, T2 = y

# parameters setting
Ea  = 72750     # activation energy J/gmol
R   = 8.314     # gas constant J/gmol/K
k0  = 7.2e10    # Arrhenius rate constant 1/min
V   = 100.0     # Volume [L]
rho = 1000.0    # Density [g/L]
Cp  = 0.239     # Heat capacity [J/g/K]
dHr = -5.0e4    # Enthalpy of reaction [J/mol]
UA  = 5.0e4     # Heat transfer [J/min/K]
q   = 100.0     # Flowrate [L/min]
cAi = 1.0       # Inlet feed concentration [mol/L]
Ti  = 350.0     # Inlet feed temperature [K]
cA0 = 0.5      # Initial concentration [mol/L]
T10 = 300.0    # Initial temperature of tank 1 [K]
T20 = 300.0     # Initial temperature of tank 2 [K]
Tc  = 300.0     # Coolant temperature [K]

def k(T):
    return k0*math.exp(-Ea/R/T)

def systemDeriv(t,y):
    # 系统状态
    # CA10， CA20， Q1，Q2 为输入
    # CA1，CA2， CB1，CB2， T1， T2为状态量
    
    CA1, T1, CB1, CA2, T2, CB2 = y
    # first
    dCA1dt = (q/V)*(CA10 - CA1) - k(T1)*CA1*CA1
    dT1dt = (q/V)*(T10-T1) - (dHr/rho/Cp)*k(T1)*CA1*CA1 + (Q1/V/rho/Cp)
    dCB1dt = -(q/V)*CB1 + k(T1)*CA1*CA1
    # second 
    dCA2dt = (q/V)*(CA20+CA1) - ((2*q)/V)*CA2 - k(T2)*CA2*CA2
    dT2dt = (q/V) * (T20+T1) - ((2*q)/V)*T2 - (dHr/rho/Cp)*k(T2)*CA2*CA2 + (Q2/V/rho/Cp)
    dCB2dt = (q/V)*CB1 - ((2*q)/V)*CB2 + k(T2)*CA2*CA2
    return [dCA1dt, dT1dt, dCB1dt, dCA2dt, dT2dt, dCB2dt]

CA10 = cAi
CA20 = cAi
CA1 = cA0
CA2 = cA0
CB1 = cA0
CB2 = cA0
Q1 = 1e3
Q2 = 1e3
T1 = T10
T2 = T20
# simulation
initialConditions = [CA1, T1, CB1, CA2, T2, CB2]
t_initial = 0.0
t_final = 10.0
t = np.linspace(t_initial, t_final/2, 2000)
soln1 = solve_ivp(systemDeriv, [t_initial, t_final/2], initialConditions, t_eval=t)

CA10 = CA10+0.5
initialConditions = [soln1.y[0][-1], soln1.y[1][-1], soln1.y[2][-1], soln1.y[3][-1], soln1.y[4][-1], soln1.y[5][-1]]
t = np.linspace(5.0, 10.0, 2000)
soln2 = solve_ivp(systemDeriv, [5.0, 10.0], initialConditions, t_eval=t)
# IC = [cA0, T10]
# t_initial = 0.0
# t_final = 10.0
# t = np.linspace(t_initial, t_final, 2000)
# soln = solve_ivp(systemDeriv, [t_initial, t_final], IC, t_eval=t)



plt.subplot(4,3,1)
plt.plot(soln1.t, soln1.y[0])
plt.subplot(4,3,2)
plt.plot(soln1.t, soln1.y[1])
plt.subplot(4,3,3)
plt.plot(soln1.t, soln1.y[2])
plt.subplot(4,3,4)
plt.plot(soln1.t, soln1.y[3])
plt.subplot(4,3,5)
plt.plot(soln1.t, soln1.y[4])
plt.subplot(4,3,6)
plt.plot(soln1.t, soln1.y[5])

plt.subplot(4,3,7)
plt.plot(soln2.t, soln2.y[0])
plt.subplot(4,3,8)
plt.plot(soln2.t, soln2.y[1])
plt.subplot(4,3,9)
plt.plot(soln2.t, soln2.y[2])
plt.subplot(4,3,10)
plt.plot(soln2.t, soln2.y[3])
plt.subplot(4,3,11)
plt.plot(soln2.t, soln2.y[4])
plt.subplot(4,3,12)
plt.plot(soln2.t, soln2.y[5])

plt.show()
print("success")
