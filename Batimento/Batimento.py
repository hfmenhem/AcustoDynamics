import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#Bolinhas de ABS 1,5mm (provavelmente de diâmetro)
#propriedades: https://www.researchgate.net/figure/Density-and-sound-speed-values-Acrylonitrile-butadiene-styrene-ABS-polylactic-acid_tbl1_329699503

Npar = 2

#Npar = 1

dicMeio = Simulador.ar(Npar)
a = np.array([[1], [1]]) #np.array(Npar*[[1.5/2]]) # [mm]
#a = np.array([[1]]) #np.array(Npar*[[1.5/2]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP

cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume


f1 = np.array(Npar*[[1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))]])
f2 = np.array(Npar*[[2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))]])

f = 40e3 #[Hz]
v0max = 14.8e3 #[mm/s]

g = np.array([0,0,-9.81e3]) #[mm/s^2]
h=0
lamb = dicMeio["c"]/f

print(f'lambda = {lamb:.2f} mm ')



r0 = np.array([[[0,0,-0.5]], [[0,0,(lamb)-0.5]]])

v0 = np.array([[[0.0,0,0]], [[0.0,0,0]]])

# r0 = np.array([[[0,0,-1]]])
# v0 = np.array([[[0.0,0,0]]])



sim = Simulador(f1, f2, f, dicMeio['c'], a, m, dicMeio['rho'], v0max, h, dicMeio['dinvis'],e=0)

dt = 1e-6
tempo = 3


rs, vs, t = sim.Simular(r0, v0, dt, tempo, g=g)
Simulador.graficos(rs, vs, t, [], a)
sim.saveSimulacao(rs, vs, t, 'data')