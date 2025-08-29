import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior


from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pickle
from scipy.optimize import minimize

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

print(f'lambda = {Lamb:.2f} mm ')

zrange0 =[-Lamb/5,Lamb/5]
zrange1 =[Lamb*((1/2)-(1/5)), Lamb*((1/2)+(1/5))]

# Dz = [Lamb*(1-(1/8)), Lamb*(1+(1/8))]
# Zranges = [Lamb*(-1), Lamb*(1)]

# #Zrangep = [-Lamb, Lamb]
# Zrangep = [Zranges[0]+Dz[0], Zranges[1]+Dz[1]]


z0, dz0 = np.linspace(*zrange0, 110,retstep=True)
z1, dz1 = np.linspace(*zrange1, 100,retstep=True)

#Força na partícula 1 (partícula 0 como source)
Fz1=np.empty((len(z0), len(z1)))
Uz1=np.empty((len(z0), len(z1)))
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
    U = sim.UGorkov(Pt, GPt) #[nJ]
    
    Fz1[i, :] = F[:,0,2]
    Uz1[i, :] = U[:,0]


#Força na partícula 0 (partícula 1 como source)
Fz0=np.empty((len(z0), len(z1)))
Uz0=np.empty((len(z0), len(z1)))
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
    U = sim.UGorkov(Pt, GPt) #[nJ]
    
    Fz0[:, i] = F[:,0,2]
    Uz0[:, i] = U[:,0]

Fz0int = RectBivariateSpline(z0, z1, Fz0)
Fz1int = RectBivariateSpline(z0, z1, Fz1)

Fz0intU = RectBivariateSpline(z0, z1, Uz0).partial_derivative(1,0)
Fz1intU = RectBivariateSpline(z0, z1, Uz1).partial_derivative(0,1)

Ut = RectBivariateSpline(z0, z1, Uz0+Uz1)

with open('força', 'rb') as dbfile:
    dado = pickle.load(dbfile)
    dbfile.close()

Fz0Map = dado[0]
Fz1Map = dado[1]

Ut(0,4)

Utl = lambda x: Ut(*x)

zmin = minimize(Utl, np.array([0, 4]))

print(zmin.x)

print(Fz0Map(*zmin.x))
print(Fz1Map(*zmin.x))

#Plotar  força no eixo z -p0
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(z0, z1, np.transpose(Fz0), levels=20)

ax.set_title("força acústica total no eixo z sobre a partícula 0")
ax.set_xlabel(r'$z_0$ [mm]')
ax.set_ylabel(r'$z_1$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")

# #Plotar  força no eixo z -p0
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(z0, z1, np.transpose(Fz0Map(z0, z1)), levels=20)

# ax.set_title("força acústica total no eixo z sobre a partícula 0 - mapa")
# ax.set_xlabel(r'$z_0$ [mm]')
# ax.set_ylabel(r'$z_1$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")


# #Plotar  força no eixo z -p0
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(z0, z1, np.transpose(-1*Fz0intU(z0, z1)), levels=20)

# ax.set_title("força acústica total (pelo potencial) no eixo z sobre a partícula 0")
# ax.set_xlabel(r'$z_0$ [mm]')
# ax.set_ylabel(r'$z_1$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")

#Plotar  força no eixo z -p1
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(z0, z1, np.transpose(Fz1), levels=20)

ax.set_title("força acústica total no eixo z sobre a partícula 1")
ax.set_xlabel(r'$z_0$ [mm]')
ax.set_ylabel(r'$z_1$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")

# #Plotar  força no eixo z -p0
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(z0, z1, np.transpose(Fz1Map(z0, z1)), levels=20)

# ax.set_title("força acústica total no eixo z sobre a partícula 1 - mapa")
# ax.set_xlabel(r'$z_0$ [mm]')
# ax.set_ylabel(r'$z_1$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")


# #Plotar  força no eixo z -p1
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(z0, z1, np.transpose(-1*Fz1intU(z0, z1)), levels=20)

# ax.set_title("força acústica total (pelo potencial) no eixo z sobre a partícula 1")
# ax.set_xlabel(r'$z_0$ [mm]')
# ax.set_ylabel(r'$z_1$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")


# #Plotar  força no eixo z -p0
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(z0, z1, np.transpose(Uz0), levels=20)

# ax.set_title("Potencial acústico total sobre a partícula 0")
# ax.set_xlabel(r'$z_0$ [mm]')
# ax.set_ylabel(r'$z_1$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$Usc z$ [nJ]', format="{x:.1e}")

# #Plotar  força no eixo z -p1
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(z0, z1, np.transpose(Uz1), levels=20)

# ax.set_title("Potencial acústico total sobre a partícula 1")
# ax.set_xlabel(r'$z_0$ [mm]')
# ax.set_ylabel(r'$z_1$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$Usc z$ [nJ]', format="{x:.1e}")


#Plotar  força no eixo z -p1
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(z0, z1, np.transpose(Uz1+Uz0), levels=20)

ax.set_title("Potencial acústico total sobre ambas partículas")
ax.set_xlabel(r'$z_0$ [mm]')
ax.set_ylabel(r'$z_1$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$Usc z$ [nJ]', format="{x:.1e}")


#Plotar  força no eixo z -p1
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(z0, z1, np.transpose(Uz0), levels=20)

ax.set_title("Potencial acústico total sobre partícula 0")
ax.set_xlabel(r'$z_0$ [mm]')
ax.set_ylabel(r'$z_1$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$Usc z$ [nJ]', format="{x:.1e}")

#Plotar  força no eixo z -p1
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(z0, z1, np.transpose(Uz1), levels=20)

ax.set_title("Potencial acústico total sobre partícula 1")
ax.set_xlabel(r'$z_0$ [mm]')
ax.set_ylabel(r'$z_1$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$Usc z$ [nJ]', format="{x:.1e}")


