import sys, os

parent_dir=os.path.dirname(__file__)
for pastas in range(2):
    parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior


from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pickle
import concurrent.futures

import time

from itertools import repeat


f=40e3 #Hz
dicMeio = Simulador.ar(1)

a = np.array([[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

v0t = 10e3 #mm/s
#v0t = 1e3 #mm/s

Lamb=dicMeio["c"]/f

h=0

sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, h, 0)

#print(f'lambda = {Lamb:.2f} mm ')

def forca1(rp0, rp1, Pin0, Pin1, GPin0,GPin1,HPin1, i):
    MR = rp1-rp0[[i],:,:]
    
    Psc = sim.PhiSc(MR, Pin0[[i],:], GPin0[[i],:,:])#Aqui não precisa transport Phin e outros porque eles estão levando como argumento rs, que já está no formato certo (ao contrário da simulação normal, em que eles levam como parâmetro rp, que precisa ser transposto)
    GPsc = sim.GradPhiSc(MR,  Pin0[[i],:], GPin0[[i],:,:])
    HPsc = sim.HPhiSc(MR, Pin0[[i],:], GPin0[[i],:,:])
    
    Pt  = Psc + Pin1
    GPt = GPsc+ GPin1
    HPt = HPsc + HPin1
    
    F = sim.FGorKov(Pt, GPt, HPt) #[uN]
    
    return F[:,0,2]

def forca0(rp0, rp1, Pin0, Pin1, GPin0,GPin1,HPin0, i):
    MR = rp0-rp1[[i],:,:]
    
    Psc = sim.PhiSc(MR, Pin1[[i],:], GPin1[[i],:,:])#Aqui não precisa transport Phin e outros porque eles estão levando como argumento rs, que já está no formato certo (ao contrário da simulação normal, em que eles levam como parâmetro rp, que precisa ser transposto)
    GPsc = sim.GradPhiSc(MR,  Pin1[[i],:], GPin1[[i],:,:])
    HPsc = sim.HPhiSc(MR, Pin1[[i],:], GPin1[[i],:,:])
    
    Pt  = Psc + Pin0
    GPt = GPsc+ GPin0
    HPt = HPsc + HPin0
    
    F = sim.FGorKov(Pt, GPt, HPt) #[uN]
    
    return F[:,0,2]

if __name__ == '__main__':
    
    start_time = time.time()
    
    nome = 'estacionaria'
    
    zrange0 =[-Lamb/5,Lamb/5]
    zrange1 =[Lamb*((1/2)-(1/5)), Lamb*((1/2)+(1/5))]
    
    z0, dz0 = np.linspace(*zrange0, 10000,retstep=True)
    z1, dz1 = np.linspace(*zrange1, 10000,retstep=True)
    

    
    rp1 =np.array([[[0,0,1]]])*np.expand_dims(z1, (1,2)) #posição das partículas de prova
    Pin1 = sim.PhiIn(rp1) #[mm^2/s]
    GPin1 = sim.GradPhiIn(rp1)
    HPin1 = sim.HPhiIn(rp1)
    
    rp0 =np.array([[[0,0,1]]])*np.expand_dims(z0, (1,2)) #posição das partículas de prova
    Pin0 = sim.PhiIn(rp0) #[mm^2/s]
    GPin0 = sim.GradPhiIn(rp0)
    HPin0 = sim.HPhiIn(rp0)
    
    
    #Força na partícula 1 (partícula 0 como source)
    Is0 = range(len(z0))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        Fz1C = executor.map(forca1, repeat(rp0), repeat(rp1), repeat(Pin0), repeat(Pin1), repeat(GPin0),repeat(GPin1),repeat(HPin1), Is0)
    
    #Força na partícula 0 (partícula 1 como source)
    Is1 = range(len(z1))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        Fz0C = executor.map(forca0, repeat(rp0), repeat(rp1), repeat(Pin0), repeat(Pin1), repeat(GPin0),repeat(GPin1),repeat(HPin0), Is1)    
    
    Fz1=np.empty((len(z0), len(z1)))
    for i,fz1 in enumerate(Fz1C):
        Fz1[i, :] = fz1
        
    Fz0=np.empty((len(z0), len(z1)))
    for i,fz0 in enumerate(Fz0C):
        Fz0[:, i] = fz0
    
    
    Fz0int = RectBivariateSpline(z0, z1, Fz0)
    Fz1int = RectBivariateSpline(z0, z1, Fz1)
            
    print(f'O código demorou {(time.time() - start_time):.2f} s')
    
    arraysalvaValores=[
                  #['deslocamento em z', f'{dz:.1f} mm'],
                  ['Partículas de PP no ar', 'usando onda plana sem arrasto'],
                  ['Amplitude de velocidade da onda plana', f'{v0t:.4e} mm/s'],
                  ['f', f'{f:.4e} Hz'],
                  ['c', f'{dicMeio["c"]:.4e} Hz'],
                  ['rho', f'{dicMeio["rho"]:.4e} Hz'],
                  ['a, todas iguais', f'{a[0,0]:.4e} mm'],
                  ['m, todas iguais', f'{m[0,0]:.4e} mm'],
                  ['rhoPP', f'{rhoPol:.4e} g/mm^3'],
                  ['cPP', f'{cPol:.4e} mm/s'],
                  ['f1', f'{f1:.4e} mm'],
                  ['f2', f'{f2:.4e} mm'],
                  ['Nodos de pressão das ondas planas', ''],
                  ['z0', f'{0:.4e} mm'],
                  ['z1', f'{Lamb/2:.4e} mm'],
                  ['amplitude de amostragem', ''],
                  ['zmin0', f'{zrange0[0]:.4e} mm'],
                  ['zmax0', f'{zrange0[1]:.4e} mm'],
                  ['zmin1', f'{zrange1[0]:.4e} mm'],
                  ['zmax1', f'{zrange1[1]:.4e} mm'],
                  ['discretização da amostragem', ''],
                  ['dz0', f'{dz0:.4e} mm'],
                  ['dz1', f'{dz1:.4e} mm'],
                  ]
    
    # arraysalvaValores=[
    #               #['deslocamento em z', f'{dz:.1f} mm'],
    #               ['Partículas de PP no ar', 'usando TiniLev sem arrasto'],
    #               ['Amplitude de velocidade do transdutor', f'{v0t:.4e} mm/s'],
    #               ['f', f'{f:.4e} Hz'],
    #               ['c', f'{dicMeio["c"]:.4e} Hz'],
    #               ['rho', f'{dicMeio["rho"]:.4e} Hz'],
    #               ['a, todas iguais', f'{a[0,0]:.4e} mm'],
    #               ['m, todas iguais', f'{m[0,0]:.4e} mm'],
    #               ['rhoPP', f'{rhoPol:.4e} g/mm^3'],
    #               ['cPP', f'{cPol:.4e} mm/s'],
    #               ['f1', f'{f1:.4e} mm'],
    #               ['f2', f'{f2:.4e} mm'],
    #               ['Nodos de pressão das ondas planas', ''],
    #               ['z0', f'{0:.4e} mm'],
    #               ['z1', f'{Lamb/2:.4e} mm'],
    #               ['amplitude de amostragem', ''],
    #               ['zmin0', f'{zrange0[0]:.4e} mm'],
    #               ['zmax0', f'{zrange0[1]:.4e} mm'],
    #               ['zmin1', f'{zrange1[0]:.4e} mm'],
    #               ['zmax1', f'{zrange1[1]:.4e} mm'],
    #               ['discretização da amostragem', ''],
    #               ['dz0', f'{dz0:.4e} mm'],
    #               ['dz1', f'{dz1:.4e} mm'],
    #               ]
    
    
    
    np.savetxt(f'{nome}-geral-força.txt', arraysalvaValores, fmt='%s')
    
    
    with open(f'{nome}-força', 'wb') as dbfile:
        pickle.dump((Fz0int, Fz1int), dbfile)
        dbfile.close()
    
    print(f'"{nome}-força" salvo!')
      
    #Plotar  força no eixo z -p0
    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    #ax.set_aspect(1)
    pcm = ax.contourf(z0, z1, Fz0, levels=20)
    
    ax.set_title("força acústica total no eixo z sobre a partícula 0")
    ax.set_xlabel(r'$z_0$ [mm]')
    ax.set_ylabel(r'$z_1$ [mm]')
    fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")
    
    #Plotar  força no eixo z -p1
    fig, ax = plt.subplots(dpi=300, figsize=(10,10))
    #ax.set_aspect(1)
    pcm = ax.contourf(z0, z1, Fz1, levels=20)
    
    ax.set_title("força acústica total no eixo z sobre a partícula 1")
    ax.set_xlabel(r'$z_0$ [mm]')
    ax.set_ylabel(r'$z_1$ [mm]')
    fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")


