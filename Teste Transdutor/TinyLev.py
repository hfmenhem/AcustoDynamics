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

sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0, 0, 0)

print(f'lambda = {Lamb:.2f} mm ')

c = sim.tinyLev(0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(c[:,0], c[:,1], c[:,2])
ax.set_aspect('equal')
plt.show()

Yrange = [-20, 20]
Zrange = [-40, 40]

ys, dy = np.linspace(*Yrange, 150,retstep=True)
zs, dz = np.linspace(*Zrange, 300,retstep=True)
z = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),0)
y = np.expand_dims([0,1,0]*np.expand_dims(ys, 1),1)

rp = (z+y).reshape(-1,1,3) #Posições das partículas de prova


Pin = sim.PhiIn(rp) #[mm^2/s]
GPin = sim.GradPhiIn(rp)
HPin = sim.HPhiIn(rp)
F = sim.FGorKov(Pin, GPin, HPin) #[uN]

PinLoc = Pin.reshape(len(ys),len(zs))
Floc = F.reshape(len(ys),len(zs),3)

nan = np.full((len(ys), len(zs)), np.nan)
Ploc = 1j*dicMeio["rho"]*(2*np.pi/Lamb)*dicMeio["c"]*PinLoc
Filtro = np.abs(Ploc)<1600

fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.set_aspect(1)

pcm = ax.pcolormesh(ys, zs, np.transpose(np.abs(Ploc))/1e3)
cbar = fig.colorbar(pcm, ax=ax, label = 'P [kPa]')
cbar.set_ticks(ticks=[0,1,2,3,4])

ax.set_title('Amplitude pressão do TinyLev (v = 1m/s)')
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
plt.show()
