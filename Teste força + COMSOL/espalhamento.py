import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Este código testa se o programa calcula corretamente a força acústica secundária



#dicMeio = Simulador.ar(1)






f1 = 1
f2 = 1
f=10*(10**3)
c = 340 * (10**3) #mm/s
k= 2*np.pi*f/c
a = np.array([[.5]]) # [mm]
rho = 1.2 * (10**-6) #g/mm^3
v0 = 1e3 #mm/s

#rhoPol = (20*(10**-6))
#cPol = 2350*(10**3) #[mm/s] 
#f1 = 1- ((rhoar*(car**2))/ (rhoPol*(cPol**2)))
#f2 = 2*((rhoPol-rhoar)/((2*rhoPol)+rhoar))





#Dados de (SIMON et al., 2019)
# f1 = 0.623
# f2 = 0.034
# f = 10*(10**6) #[1/s]
# c = 1480*(10**3) #[mm/s]
# k = 2*np.pi*f/c #[1/mm]
# a = np.array([[0.1*np.pi/k]]) # [mm]
# #m = (a**3*(4*np.pi/3))*(1*10**3) # [g], massa, não importa
# rho = 998*(10**(-6)) #g/mm^3
# #v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
# #Vo = P0*k/omega*Rho
# v0 = (50*(10**3))*k/(2*np.pi*f*rho) #Pressão = 50*(10**3) Pa = 50*(10**3) g/mm*s^2



Lamb=c/f

h=0
#h=(np.pi/2)/k
m = np.zeros(np.shape(a)) # [g], densidade do ar vezes seu volume

sim = Simulador(np.array([[f1]]), np.array([[f2]]),f, c, a, m,rho, v0, h, 0)
rs = np.array([[[0,0,0]]]) #Posições das partículas emissoreas

print(f'Z max = {Lamb} mm ')

Yrange = [0, Lamb]
Zrange = [-Lamb, Lamb]

ys, dy = np.linspace(*Yrange, 200,retstep=True)
zs, dz = np.linspace(*Zrange, 400,retstep=True)

py = np.array([0,1,0])*np.expand_dims(ys, 1)
pz = np.array([0,0,1])*np.expand_dims(zs, 1)
rp = np.reshape(np.concatenate((py,pz)), (-1,1,3))


MR = rp-rs
Pin = sim.PhiIn(rp) #[mm^2/s]
GPin = sim.GradPhiIn(rp)
HPin = sim.HPhiIn(rp)

Psc = sim.PhiSc(MR, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))
GPsc = sim.GradPhiSc(MR, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))
HPsc = sim.HPhiSc(MR, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))

Pt  = np.sum(Psc , axis = 1,keepdims=True) + Pin
GPt = np.sum(GPsc, axis = 1,keepdims=True) + GPin
HPt = np.sum(HPsc, axis = 1,keepdims=True) + HPin

# Pt  = Pin
# GPt = GPin
# HPt = HPin

F = sim.FGorKov(Pt, GPt, HPt) #[uN]
Fin = sim.FGorKov(Pin, GPin, HPin)
Fsc = F-Fin

Fscy = Fsc[:len(py),0, 1]
Fscz = Fsc[len(py):,0, 2]
Fty = F[:len(py),0,1]
Ftz = F[len(py):,0,2]




#filtro:
Filtroy = ys >3.5
Filtroz = zs>1

# Filtroy = ys >.01
# Filtroz = zs>0.01

Fscy = np.where(Filtroy,Fscy, np.full(np.shape(Fscy), np.nan))
Fscz = np.where(Filtroz,Fscz, np.full(np.shape(Fscz), np.nan))
Fty = np.where(Filtroy,Fty, np.full(np.shape(Fty), np.nan))
Ftz =  np.where(Filtroz,Ftz, np.full(np.shape(Ftz), np.nan))


dataZcomsol = pd.read_csv('Total_force_axial.txt', sep=",", header=None, skiprows=8)
zcomsol= np.array(dataZcomsol[0])
Fzcomsol=np.array(dataZcomsol[1])*1e6

dataYcomsol = pd.read_csv('Total_force_lateral.txt', sep=",", header=None, skiprows=8)
ycomsol= np.array(dataYcomsol[0])
Fycomsol=np.array(dataYcomsol[1])*1e6

#Plotar módulo da força
figy, axy = plt.subplots(dpi=300, figsize=(10,10))
plt.plot(ys, Fty, label='Python')
plt.plot(ycomsol, Fycomsol, label='COMSOL')
axy.set_title("Força acústica total no eixo y - h=0")
axy.set_xlabel('y [mm]')
axy.set_ylabel('Fy [nN]')
plt.legend(fontsize=12)
figy.show()

figy, axy = plt.subplots(dpi=300, figsize=(10,10))
plt.plot(zs, Ftz, label='Python')
plt.plot(zcomsol, Fzcomsol, label='COMSOL')
axy.set_title("Força acústica total no eixo z - h=0")
axy.set_xlabel('z [mm]')
axy.set_ylabel('Fz [uN]')
plt.legend(fontsize=12)
figy.show()

