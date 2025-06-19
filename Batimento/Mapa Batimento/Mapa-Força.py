import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior


from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pickle

#nome = 'estacionaria'
numeroSim='Sim9'

f=40e3 #Hz
dicMeio = Simulador.ar(1)

a = np.array([[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

#v0t = 10e3 #mm/s
v0t = 1e3 #mm/s

Lamb=dicMeio["c"]/f

h=0

sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, h, 0)
fase= np.pi
sim.tinyLev(fase)

print(f'lambda = {Lamb:.2f} mm ')

zrange0 =[-Lamb/5,Lamb/5]
zrange1 =[Lamb*((1/2)-(1/5)), Lamb*((1/2)+(1/5))]

# Dz = [Lamb*(1-(1/8)), Lamb*(1+(1/8))]
# Zranges = [Lamb*(-1), Lamb*(1)]

# #Zrangep = [-Lamb, Lamb]
# Zrangep = [Zranges[0]+Dz[0], Zranges[1]+Dz[1]]


z0, dz0 = np.linspace(*zrange0, 1000,retstep=True)
z1, dz1 = np.linspace(*zrange1, 1000,retstep=True)

#Força na partícula 1 (partícula 0 como source)
Fz1=np.empty((len(z0), len(z1)))
for i, zsi in enumerate(z0):
    
    rp =np.array([[[0,0,1]]])*np.expand_dims(z1, (1,2)) #posição das partículas de prova
    rs =np.array([[[0,0,1]]])*np.expand_dims([z0[i]], (0,2)) #posição das partícula espalhadora
    MR = rp-rs
    Pin = sim.PhiIn(rp) #[mm^2/s]
    GPin = sim.GradPhiIn(rp)
    HPin = sim.HPhiIn(rp)
    
    Psc = sim.PhiSc(MR, sim.PhiIn(rs), sim.GradPhiIn(rs))#Aqui não precisa transport Phin e outros porque eles estão levando como argumento rs, que já está no formato certo (ao contrário da simulação normal, em que eles levam como parâmetro rp, que precisa ser transposto)
    GPsc = sim.GradPhiSc(MR,  sim.PhiIn(rs), sim.GradPhiIn(rs))
    HPsc = sim.HPhiSc(MR, sim.PhiIn(rs), sim.GradPhiIn(rs))
    
    Pt  = Psc + Pin
    GPt = GPsc+ GPin
    HPt = HPsc + HPin
    
    F = sim.FGorKov(Pt, GPt, HPt) #[uN]
    
    Fz1[i, :] = F[:,0,2]


#Força na partícula 0 (partícula 1 como source)
Fz0=np.empty((len(z0), len(z1)))
for i, zsi in enumerate(z1):
    
    rp =np.array([[[0,0,1]]])*np.expand_dims(z0, (1,2)) #posição das partículas de prova
    rs =np.array([[[0,0,1]]])*np.expand_dims([z1[i]], (0,2)) #posição das partícula espalhadora
    MR = rp-rs
    Pin = sim.PhiIn(rp) #[mm^2/s]
    GPin = sim.GradPhiIn(rp)
    HPin = sim.HPhiIn(rp)
    
    Psc = sim.PhiSc(MR, sim.PhiIn(rs), sim.GradPhiIn(rs))#Aqui não precisa transport Phin e outros porque eles estão levando como argumento rs, que já está no formato certo (ao contrário da simulação normal, em que eles levam como parâmetro rp, que precisa ser transposto)
    GPsc = sim.GradPhiSc(MR,  sim.PhiIn(rs), sim.GradPhiIn(rs))
    HPsc = sim.HPhiSc(MR, sim.PhiIn(rs), sim.GradPhiIn(rs))
    
    Pt  = Psc + Pin
    GPt = GPsc+ GPin
    HPt = HPsc + HPin
    
    F = sim.FGorKov(Pt, GPt, HPt) #[uN]
    
    Fz0[:, i] = F[:,0,2]


Fz0int = RectBivariateSpline(z0, z1, Fz0)
Fz1int = RectBivariateSpline(z0, z1, Fz1)


if os.path.exists(f'{numeroSim}'):
    versao = 2
    while(os.path.exists(f'{numeroSim}-v{versao}')):
        versao+=1
    diretorio = f'{numeroSim}-v{versao}'
else:
    diretorio = f'{numeroSim}'
        

os.mkdir(diretorio)
# arraysalvaValores=[
#               #['deslocamento em z', f'{dz:.1f} mm'],
#               ['Partículas de PP no ar', 'usando onda plana sem arrasto'],
#               ['Amplitude de velocidade da onda plana', f'{v0t:.4e} mm/s'],
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

arraysalvaValores=[
              #['deslocamento em z', f'{dz:.1f} mm'],
              ['Partículas de PP no ar', 'usando TiniLev sem arrasto'],
              ['Amplitude de velocidade do transdutor', f'{v0t:.4e} mm/s'],
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



np.savetxt(f'{diretorio}\\{numeroSim}-geral-força.txt', arraysalvaValores, fmt='%s')


with open(f'{diretorio}\\força', 'wb') as dbfile:
    pickle.dump((Fz0int, Fz1int), dbfile)
    dbfile.close()

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


