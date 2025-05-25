import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

import scipy as sc

#Este código testa se o programa calcula corretamente o potencial de velocidade primário de um transdutor



def plotgrafico(numerico, simulado, legenda):
    nan = np.full((len(ys), len(zs)), np.nan)
    inf = np.full((len(ys), len(zs)), np.inf)
    zero = np.full((len(ys), len(zs)), 0)
   
    normR = mpl.colors.Normalize(vmin=np.min(np.real([np.where(filtro, numerico, inf), np.where(filtro, simulado, inf)])), vmax=np.max(np.real([np.where(filtro, numerico, -1*inf), np.where(filtro, simulado, -1*inf)])))
    normI = mpl.colors.Normalize(vmin=np.min(np.imag([np.where(filtro, numerico, inf), np.where(filtro, simulado, inf)])), vmax=np.max(np.imag([np.where(filtro, numerico, -1*inf), np.where(filtro, simulado, -1*inf)])))
    normA = mpl.colors.Normalize(vmin=np.min(np.abs([np.where(filtro, numerico, inf), np.where(filtro, simulado, inf)])), vmax=np.max(np.abs([np.where(filtro, numerico, zero), np.where(filtro, simulado, zero)])))
 
    
    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.real(np.where(filtro, numerico, nan))), norm=normR)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} numérico, parte real')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()

    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.real(np.where(filtro, simulado, nan))), norm=normR)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} simulado, parte real')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()

    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.imag(np.where(filtro, numerico, nan))), norm=normI)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} numérico, parte imaginária')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()
   
    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.imag(np.where(filtro, simulado, nan))), norm=normI)
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} simulado, parte imaginária')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()
    
    
    # fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    # ax.set_aspect(1)
    # pcm = ax.pcolormesh(ys, zs, np.transpose(np.abs(np.where(filtro, numerico, nan))), norm=normA)
    # fig.colorbar(pcm, ax=ax)
    # ax.set_title(f'{legenda} numérico, valor absoluto')
    # ax.set_xlabel('y [mm]')
    # ax.set_ylabel('z [mm]')
    # fig.show()
   
    # fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    # ax.set_aspect(1)
    # pcm = ax.pcolormesh(ys, zs, np.transpose(np.abs(np.where(filtro, simulado, nan))), norm=normA)
    # fig.colorbar(pcm, ax=ax)
    # ax.set_title(f'{legenda} simulado, valor absoluto')
    # ax.set_xlabel('y [mm]')
    # ax.set_ylabel('z [mm]')
    # fig.show()

    #plt.close()


f=40e3 #Hz
dicMeio = Simulador.ar(1)
pressao = 800#Pa = g/mm*s^2
#pressao = 2000#Pa = g/mm*s^2

a = np.array([[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

v0 = 1e3#mm/s

Lamb=dicMeio["c"]/f


h=0
h=Lamb/3

m = np.zeros(np.shape(a)) # [g], densidade do ar vezes seu volume

sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0, h, 0)

L = 100
r0 = np.array([[0,0,0], [0,0, L]])
n = np.array([[0,0,1], [0,0,-1]])
raio = np.array([5,5])

sim.setTransdutor(r0, n, raio, fase=[0, 0])


print(f'Z max = {Lamb} mm ')

Yrange = [-20, 20]
Zrange = [L/4, 3*L/4]

ys, dy = np.linspace(*Yrange, 400,retstep=True)
zs, dz = np.linspace(*Zrange, 400,retstep=True)
z = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),0)
y = np.expand_dims([0,1,0]*np.expand_dims(ys, 1),1)

rp = (z+y).reshape(-1,1,3) #Posições das partículas de prova


Pin = sim.PhiIn(rp) #[mm^2/s]
GPin = sim.GradPhiIn(rp)
HPin = sim.HPhiIn(rp)
F = sim.FGorKov(Pin, GPin, HPin) #[uN]


PinLoc = Pin.reshape(len(ys),len(zs))
rploc = rp.reshape(len(ys),len(zs),3)
GPinloc = GPin.reshape(len(ys),len(zs),3)
Floc = F.reshape(len(ys),len(zs),3)

HPinloc = HPin.reshape(len(ys),len(zs),3,3)


gradNumPin = np.gradient(PinLoc, ys, zs)
#HnumPscy =  np.gradient(gradNumPsc[0], ys, zs)
#HnumPscz =  np.gradient(gradNumPsc[1], ys, zs)
HnumPiny =  np.gradient(GPinloc[:,:,1], ys, zs)
HnumPinz =  np.gradient(GPinloc[:,:,2], ys, zs)

filtro = np.full(np.shape(PinLoc), True)
filtro = np.linalg.norm(rploc, axis=2)>raio[0]

nan = np.full((len(ys), len(zs)), np.nan)
Ploc = 1j*dicMeio["rho"]*(2*np.pi/Lamb)*dicMeio["c"]*PinLoc
Filtro = np.abs(Ploc)<800


fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.set_aspect(1)

pcm = ax.contourf(ys, zs, np.transpose(np.abs(np.where(Filtro,Ploc, nan))), levels=100)
fig.colorbar(pcm, ax=ax, label = 'Pin, [Pa]')
ax.set_title('pressão do transdutor, valor absoluto')
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
fig.show()


# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# plt.plot(zs, np.abs(np.where(Filtro,Ploc, nan))[0,:])
# plt.grid()

# plotgrafico(gradNumPin[0], GPinloc[:,:,1], 'd Psc / dy')
# plotgrafico(gradNumPin[1], GPinloc[:,:,2], 'd Psc / dz')


#plotgrafico(HnumPiny[0], HPinloc[:,:,1,1], 'd^2 Psc / dydy')
plotgrafico(HnumPiny[1], HPinloc[:,:,1,2], 'd^2 Psc / dzdy')
#plotgrafico(HnumPinz[0], HPinloc[:,:,2,1], 'd^2 Psc / dydz')
#plotgrafico(HnumPinz[1], HPinloc[:,:,2,2], 'd^2 Psc / dzdz')

fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.set_aspect(1)
pcm = ax.pcolormesh(ys, zs, np.transpose(np.linalg.norm(Floc, axis=2)))
fig.colorbar(pcm, ax=ax,  label='|Fin| [uN]')
ax.set_title(f'Módulo da força acústica primária')
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
fig.show()

plt.show()



