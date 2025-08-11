import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#Este código testa se o programa calcula corretamente a força acústica secundária


f=240 #Hz
dicMeio = Simulador.ar(1)
pressao = 500#Pa = g/mm*s^2
#pressao = 2000#Pa = g/mm*s^2

a = np.array([[3]]) # [mm]
rhoPol = (20*(10**-6))
cPol = 2350*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

g =np.array( [0,-9.81e3, 0]) * (1-(dicMeio['rho']/rhoPol))#g efetivo considerando empuxo

Lamb=dicMeio["c"]/f



#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
#Vo = P0*k/omega*Rho
v0 = pressao/(dicMeio['rho']*dicMeio['c'])

h=0

sim = Simulador(np.array([[f1]]), np.array([[f2]]),f, dicMeio['c'], a, m, dicMeio['rho'], v0, h, 0)
rs = np.array([[[0,0,0]]]) #Posições das partículas emissoras

print(f'Z max = {Lamb} mm ')

Yrange = [0, Lamb/20]
Zrange = [-Lamb/20, Lamb/20]

Yrange = [0, 20]
Zrange = [-20, 20]

# Yrange = [5, 20]
# Zrange = [5, 20]

ys, dy = np.linspace(*Yrange, 200,retstep=True)
zs, dz = np.linspace(*Zrange, 400,retstep=True)
z = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),0)
y = np.expand_dims([0,1,0]*np.expand_dims(ys, 1),1)

rp = (z+y).reshape(-1,1,3) #Posições das partículas de prova

MR = rp-rs
PinB = sim.PhiIn(rp) #[mm^2/s]
GPinB = sim.GradPhiIn(rp)

PinA = sim.PhiIn(rs) #[mm^2/s]
GPinA = sim.GradPhiIn(rs)

PscB = sim.PhiSc(MR, np.transpose(PinA), np.transpose(GPinA, axes=(1,0,2)))
GPscB = sim.GradPhiSc(MR, np.transpose(PinA), np.transpose(GPinA, axes=(1,0,2)))

PscA = sim.PhiSc(np.transpose(MR, axes = (1,0,2)), np.transpose(PinB), np.transpose(GPinB, axes=(1,0,2)))
GPscA = sim.GradPhiSc(np.transpose(MR, axes = (1,0,2)), np.transpose(PinB), np.transpose(GPinB, axes=(1,0,2)))

PtB  = PscB  + PinB
GPtB = GPscB + GPinB

PtA  = PscA  + PinA
GPtA = GPscA + GPinA

UB = sim.UGorkov(PtB, GPtB)
UA = np.transpose(sim.UGorkov(PtA, GPtA))
UT = (UA+UB).reshape(len(ys),(len(zs)))

Ug =-1* m[0,0]*np.einsum('ijk, k -> ij', rp, g).reshape(len(ys),len(zs))

UT = UT + Ug

#Plotar o potencial
#filtro:
Filtro = np.linalg.norm(rp-rs, axis=2)>a[0,0]
nan = np.empty((200,400))
nan[:,:]= np.nan

U2F = np.where(np.reshape(Filtro, (200,400)),UT, nan)


fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.set_aspect(1)
pcm = ax.contourf(ys, zs, np.transpose(U2F),levels=40)

fig.colorbar(pcm, ax=ax, label='Ut [J]')
ax.set_title(f'Potencial total - $x_A = ${rs[0,0,0]:.1f} mm')
ax.set_xlabel('$z_B$ [mm]')
ax.set_ylabel('$x_B$ [mm]')
fig.show()

