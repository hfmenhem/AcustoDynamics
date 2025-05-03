import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

Npar = 5
f=120 #Hz
dicMeio = Simulador.ar(Npar)
pressao = 800#Pa = g/mm*s^2

a = np.array(Npar*[[5]]) # [mm]
rhoPol = (20*(10**-6))
cPol = 2350*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume
m = np.array([[.1], [.3], [.3], [0.3], [0.3]])


f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

print(f'f1 = {f1:.4f}')
print(f'f2 = {f2:.4f}')
print(f'lambda = { dicMeio["c"]/f} mm ')
h=0


r0 = np.array([[[0,0,0]], [[-20,0,0]], [[20,0,0]], [[0,0,-20]], [[0,0,20]]])
v0 = np.array(Npar*[[[0.0,0,0]]])

sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), 20*(10**3), dicMeio['c'], a, m, dicMeio['rho'], pressao/(dicMeio['rho']*dicMeio['c']), h, dicMeio['dinvis'])
dt = 0.001
tempo = .5

rs, vs, t, TColsisoes = sim.SimularComColisão(r0, v0, dt, tempo)
Simulador.graficos(rs, vs, t, TColsisoes, a)
sim.saveSimulacao(rs, vs, t, f'Tubo1-{pressao}')