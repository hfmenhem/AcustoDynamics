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



# f1 = 1
# f2 = 1
# f=10*(10**3)
# c = 340 * (10**3) #mm/s
# k= 2*np.pi*f/c
# a = np.array([[.5]]) # [mm]
# rho = 1.2 * (10**-6) #g/mm^3
# v0 = 1e3 #mm/s

#rhoPol = (20*(10**-6))
#cPol = 2350*(10**3) #[mm/s] 
#f1 = 1- ((rhoar*(car**2))/ (rhoPol*(cPol**2)))
#f2 = 2*((rhoPol-rhoar)/((2*rhoPol)+rhoar))

# #Dados de (SIMON et al., 2019) - ÁGUA
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




#Dados de (SIMON et al., 2019) - AR (MODIFICADO)
#f1 = 0.99998
#f2 = 0.99825
f1 = 0
f2 = 1
f = 40*(10**3) #[1/s]
c = 343*(10**3) #[mm/s]
k = 2*np.pi*f/c #[1/mm]
Lamb = c/f
a = np.array([[Lamb/20]]) # [mm]
#m = (a**3*(4*np.pi/3))*(1*10**3) # [g], massa, não importa
rho = 1.225*(10**(-6)) #g/mm^3
#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
#Vo = P0*k/omega*Rho
v0 = (50*(10**3))*k/(2*np.pi*f*rho) #Pressão = 50*(10**3) Pa = 50*(10**3) g/mm*s^2
v0 = 1e3 #1m/s


h=0
h=Lamb/3

m = np.zeros(np.shape(a)) # [g], densidade do ar vezes seu volume

sim = Simulador(np.array([[f1]]), np.array([[f2]]),f, c, a, m,rho, v0, h, 0)

L = 100
r0 = np.array([[0,0,0], [0,0, L]])
n = np.array([[0,0,1], [0,0,-1]])
raio = np.array([5,5])

sim.setTransdutor(r0, n, raio)


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
Ploc = 1j*rho*k*c*PinLoc
Filtro = np.abs(Ploc)<800


fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.set_aspect(1)

pcm = ax.contourf(ys, zs, np.transpose(np.abs(np.where(Filtro,Ploc, nan))), levels=100)
fig.colorbar(pcm, ax=ax)
ax.set_title('$\phi$ do transdutor, valor absoluto')
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
fig.colorbar(pcm, ax=ax,  label='|Fsc| [uN]')
ax.set_title(f'Módulo da força acústica primária')
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
fig.show()

plt.show()



