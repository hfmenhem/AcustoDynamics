import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Este código testa se o programa calcula corretamente a força acústica secundária



#dicMeio = Simulador.ar(1)


def plotgrafico(numerico, simulado, legenda):
    
    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    nan = np.full((len(ys), len(zs)), np.nan)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.real(np.where(filtro, numerico, nan))))
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} numérico, parte real')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()

    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.real(np.where(filtro, simulado, nan))))
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} simulado, parte real')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()

    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.imag(np.where(filtro, numerico, nan))))
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} numérico, parte imaginária')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()
   
    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    ax.set_aspect(1)
    pcm = ax.pcolormesh(ys, zs, np.transpose(np.imag(np.where(filtro, simulado, nan))))
    fig.colorbar(pcm, ax=ax)
    ax.set_title(f'{legenda} simulado, parte imaginária')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('z [mm]')
    fig.show()

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
f = 10*(10**3) #[1/s]
c = 343*(10**3) #[mm/s]
k = 2*np.pi*f/c #[1/mm]
Lamb = c/f
a = np.array([[Lamb/20]]) # [mm]
#m = (a**3*(4*np.pi/3))*(1*10**3) # [g], massa, não importa
rho = 1.225*(10**(-6)) #g/mm^3
#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
#Vo = P0*k/omega*Rho
v0 = (50*(10**3))*k/(2*np.pi*f*rho) #Pressão = 50*(10**3) Pa = 50*(10**3) g/mm*s^2


h=0
h=Lamb/3

m = np.zeros(np.shape(a)) # [g], densidade do ar vezes seu volume

sim = Simulador(np.array([[f1]]), np.array([[f2]]),f, c, a, m,rho, v0, h, 0)
rs = np.array([[[0,0,0]]]) #Posições das partículas emissoreas

print(f'Z max = {Lamb} mm ')

Yrange = [0, Lamb]
Zrange = [-Lamb, Lamb]

ys, dy = np.linspace(*Yrange, 200,retstep=True)
zs, dz = np.linspace(*Zrange, 400,retstep=True)
z = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),0)
y = np.expand_dims([0,1,0]*np.expand_dims(ys, 1),1)

rp = (z+y).reshape(-1,1,3) #Posições das partículas de prova

MR = rp-rs
Pin = sim.PhiIn(rp) #[mm^2/s]
GPin = sim.GradPhiIn(rp)
HPin = sim.HPhiIn(rp)

Psc = sim.PhiSc(MR, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))
GPsc = sim.GradPhiSc(MR, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))
HPsc = sim.HPhiSc(MR, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))

PscLoc = Psc.reshape(len(ys),len(zs))
rploc = rp.reshape(len(ys),len(zs),3)
GPscloc = GPsc.reshape(len(ys),len(zs),3)
HPscloc = HPsc.reshape(len(ys),len(zs),3,3)


gradNumPsc = np.gradient(PscLoc, ys, zs)
#HnumPscy =  np.gradient(gradNumPsc[0], ys, zs)
#HnumPscz =  np.gradient(gradNumPsc[1], ys, zs)
HnumPscy =  np.gradient(GPscloc[:,:,1], ys, zs)
HnumPscz =  np.gradient(GPscloc[:,:,2], ys, zs)

filtro = np.linalg.norm(rploc, axis=2)>a[0,0]

#as primeiras derivadas estão certas!
#plotgrafico(gradNumPsc[0], GPscloc[:,:,1], 'd Psc / dy')
#plotgrafico(gradNumPsc[1], GPscloc[:,:,2], 'd Psc / dz')


plotgrafico(HnumPscy[0], HPscloc[:,:,1,1], 'd^2 Psc / dydy')
plotgrafico(HnumPscy[1], HPscloc[:,:,1,2], 'd^2 Psc / dzdy')
plotgrafico(HnumPscz[0], HPscloc[:,:,2,1], 'd^2 Psc / dydz')
plotgrafico(HnumPscz[1], HPscloc[:,:,2,2], 'd^2 Psc / dzdz')

plt.show()



