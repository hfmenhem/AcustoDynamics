import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

Npar = 2
f=240 #Hz
#
dicMeio = Simulador.ar(Npar)
pressao = 800#Pa = g/mm*s^2

a = np.array(Npar*[[3]]) # [mm]
rhoPol = (20*(10**-6))
cPol = 2350*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume


f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

print(f'f1 = {f1:.4f}')
print(f'f2 = {f2:.4f}')
print(f'lambda = { dicMeio["c"]/f} mm ')
h=0


#r0 = np.array([[[0,0,0]], [[-20,0,0]], [[20,0,0]], [[0,0,-20]], [[0,0,20]]])
#r0 = np.array([[[10,0,0]], [[10,0,20]], [[10,0,-20]], [[20,0,-10]], [[20,0,10]]])
#r0 = np.array([[[10,0,0]], [[20,0,1]], [[30,0,1.2]], [[40,0,1]], [[50,0,0.8]], [[60,0,0.7]]])
r0 = np.array([[[10,0,0]], [[30,0,1]]])
#r0 = np.array([[[0,0,0]], [[0,0,20]], [[0,0,-20]]])
v0 = np.array(Npar*[[[0.0,0,0]]])

sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], pressao/(dicMeio['rho']*dicMeio['c']), h, dicMeio['dinvis'], plano=[[1,0,0], [0,0,0]], e=0.0)
dt = 0.01
tempo = 1

rs, vs, t, TColsisoes = sim.SimularComColisão3(r0, v0, dt, tempo, g=[-9.81 * (10**3),0,0])
Simulador.graficos(rs, vs, t, TColsisoes, a)
sim.saveSimulacao(rs, vs, t, f'Tubo1-{pressao}')