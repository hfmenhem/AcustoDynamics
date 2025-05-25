import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt

#Este código testa se o programa calcula corretamente a força acústica primária. 
#Para isso, é usado a formula padrão do potencial de Gor'kov (Uteste), 
#que é derivado numericamente e plotado junto com a força calculada pelo programa, 
#inserindo apenas Pin, Gin e Hin na função "FGorKov".


f=40e3 #Hz
dicMeio = Simulador.ar(1)


a = np.array([[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

v0 = 1 #mm/s

Lamb=dicMeio["c"]/f

h=0
sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0, h, 0)

L=100

r0 = np.array([[0,0,0], [0,0,L]])
n = np.array([[0,0,1], [0,0,-1]])
raio = np.array([5,5])
# r0 = np.array([[0,0,0]])
# n = np.array([[0,0,1]])
# raio = np.array([5])


sim.setTransdutor(r0, n, raio)


zs = np.linspace(2*raio[0], L-(2*raio[0]), 1000)
r = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),1)

Pin = sim.PhiIn(r) #[mm^2/s]
GPin = sim.GradPhiIn(r)
HPin = sim.HPhiIn(r)
F = sim.FGorKov(Pin, GPin, HPin) #[uN]


plt.figure(dpi=300)
plt.plot(r[:, 0, 2],F[:, 0, 2], linestyle='-', marker='')


plt.show()










