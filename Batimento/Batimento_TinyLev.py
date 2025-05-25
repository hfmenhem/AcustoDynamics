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

dt = 1e-6
tcar = m[0,0]/(dicMeio['dinvis']*1e3)#tempo característico devido à força de arrasto

#Partícula no nodo central
r0 = np.array([[[0.0,0,0]]])
v0 = np.array([[[0.0,0,0]]])

rs, vs, t = sim.Simular(r0, v0, dt, tcar, g=g)

while not np.all(np.isclose(vs[0,-1, :],0)): #Simula até que a velocidade seja nula
    rs, vs, t = sim.Simular(rs[:,-1,:], vs[:,-1,:], dt, tcar, g=g)
    
req0 = rs[0,-1,:]

#partícula no nodo logo acima do central

r0 = np.array([[[0.0,0,Lamb/2]]])
v0 = np.array([[[0.0,0,0]]])

rs, vs, t = sim.Simular(r0, v0, dt, tcar, g=g)

while not np.all(np.isclose(vs[0,-1, :],0)): #Simula até que a velocidade seja nula
    rs, vs, t = sim.Simular(rs[:,-1,:], vs[:,-1,:], dt, tcar, g=g)
    
req1 = rs[0,-1,:]

#2 partículas
r0 = np.array([[[0.0,0,0]],[[0.0,0,Lamb/2]]])
v0 = np.array([[[0.0,0,0]],[[0.0,0,0]]])

rs, vs, t = sim.Simular(r0, v0, dt, tcar, g=g)

while not np.all(np.isclose(vs[:,-1, :],0)): #Simula até que a velocidade seja nula
    rs, vs, t = sim.Simular(rs[:,-1,:], vs[:,-1,:], dt, tcar, g=g)
    
req01 = rs[:,-1,:]



print('nodos de pressão, se fossem ondas planas')
print(f'r0 = {0:.2f} mm')
print(f'r1 = {Lamb/2:.2f} mm')

print('pontos de equilíbrio, partículas isoladas')
print(f'r0 = {req0[2]:.2f} mm')
print(f'r1 = {req1[2]:.2f} mm')

print('pontos de equilíbrio, partículas simultâneas')
print(f'r0 = {req01[0,2]:.2f} mm')
print(f'r1 = {req01[1,2]:.2f} mm')


#Simulador.graficos(rs, vs, t, [], a)
#sim.saveSimulacao(rs, vs, t, 'data')