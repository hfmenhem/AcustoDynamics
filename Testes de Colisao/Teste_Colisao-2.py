import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

Npar = 2

dicMeio = Simulador.ar(Npar)
teste=dicMeio['k']
a = np.array(Npar*[[0.1*np.pi/dicMeio['k']]]) # [mm]
m = (a**3*(4*np.pi/3))*(dicMeio['rho']*1000) # [g], densidade do ar vezes seu volume

h=0


r0 = np.array([[[0.0,0,-0.13*np.pi/dicMeio['k']]], [[0,0,+0.13*np.pi/dicMeio['k']]]])
v0 = np.array([[[0.0,0,0]], [[0.0,0,0]]])


sim = Simulador(dicMeio['f1'], dicMeio['f2'], dicMeio['f'], dicMeio['c'], a, m, dicMeio['rho'], dicMeio['v0']/10, h, dicMeio['dinvis'])
dt = 0.001
tempo = 10

rs, vs, t, TColsisoes = sim.SimularComColisão(r0, v0, dt, tempo)

Simulador.graficos(rs, vs, t, TColsisoes, a)

sim.saveSimulacao(rs, vs, t, 'data2')