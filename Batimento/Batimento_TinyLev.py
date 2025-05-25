import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

f=40e3 #Hz
dicMeio = Simulador.ar(1)


a = np.array([[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume
f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))
v0 = 1e3 #m/s
Lamb=dicMeio["c"]/f

print(f'lambda = {Lamb:.2f} mm ')

sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0, 0, 1e3*dicMeio['dinvis'])
sim.tinyLev(np.pi)

g=[0,0,-9.81e3]

#Pontos de equilíbrio 1 partícula

Npar = 1
r0 = np.array([[[0.0,0,0]]])
v0 = np.array([[[0.0,0,0]]])

dt = 1e-6
tcar = m[0,0]/(dicMeio['dinvis']*1e3)#tempo característico devido à força de arrasto

rs, vs, t = sim.Simular(r0, v0, dt, tcar, g=g)
Simulador.graficos(rs, vs, t, [], a)
sim.saveSimulacao(rs, vs, t, 'data')