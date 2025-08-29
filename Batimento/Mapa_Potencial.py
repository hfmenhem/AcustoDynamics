import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior


from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pickle


f=40e3 #Hz
dicMeio = Simulador.ar(1)

a = np.array([[1]]) # [mm]
a = np.array([[.795]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

print(m)

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

g = 9.81e3
g=0
#g=0

v0t = 10e3 #mm/s]

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


z0, dz0 = np.linspace(*zrange0, 300,retstep=True)
z1, dz1 = np.linspace(*zrange1, 300,retstep=True)

# z0, dz0 = np.linspace(*zrange0, 1000,retstep=True)
# z1, dz1 = np.linspace(*zrange1, 1000,retstep=True)


#Força na partícula 1 (partícula 0 como source)
U1=np.empty((len(z0), len(z1)))
for i, zsi in enumerate(z0):
    
    rp =np.array([[[0,0,1]]])*np.expand_dims(z1, (1,2)) #posição das partículas de prova
    rs =np.array([[[0,0,1]]])*np.expand_dims([z0[i]], (0,2)) #posição das partícula espalhadora
    MR = rp-rs
    Pin = sim.PhiIn(rp) #[mm^2/s]
    GPin = sim.GradPhiIn(rp)

    Psc = sim.PhiSc(MR, sim.PhiIn(rs), sim.GradPhiIn(rs))#Aqui não precisa transport Phin e outros porque eles estão levando como argumento rs, que já está no formato certo (ao contrário da simulação normal, em que eles levam como parâmetro rp, que precisa ser transposto)
    GPsc = sim.GradPhiSc(MR,  sim.PhiIn(rs), sim.GradPhiIn(rs))

    Pt  = Psc + Pin
    GPt = GPsc+ GPin

    U = sim.UGorkov(Pt, GPt) #[uN]
    
    U1[i, :] = U[:,0]


#Força na partícula 0 (partícula 1 como source)
U0=np.empty((len(z0), len(z1)))
for i, zsi in enumerate(z1):
    
    rp =np.array([[[0,0,1]]])*np.expand_dims(z0, (1,2)) #posição das partículas de prova
    rs =np.array([[[0,0,1]]])*np.expand_dims([z1[i]], (0,2)) #posição das partícula espalhadora
    MR = rp-rs
    Pin = sim.PhiIn(rp) #[mm^2/s]
    GPin = sim.GradPhiIn(rp)
    
    Psc = sim.PhiSc(MR, sim.PhiIn(rs), sim.GradPhiIn(rs))#Aqui não precisa transport Phin e outros porque eles estão levando como argumento rs, que já está no formato certo (ao contrário da simulação normal, em que eles levam como parâmetro rp, que precisa ser transposto)
    GPsc = sim.GradPhiSc(MR,  sim.PhiIn(rs), sim.GradPhiIn(rs))
    
    Pt  = Psc + Pin
    GPt = GPsc+ GPin
    
    U = sim.UGorkov(Pt, GPt) #[uN]
    
    U0[:, i] = U[:,0]

Ug = g*m[0,0]*(np.expand_dims(z1, 0)+np.expand_dims(z0, 1))

Ut = (((U1+U0))+Ug)

fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(z0, z1, np.transpose(Ut), levels=20)
ax.set_title("Potencial acústico total do sistema")
ax.set_xlabel(r'$z_0$ [mm]')
ax.set_ylabel(r'$z_1$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$U_t$ []', format="{x:.1e}")

nome = 'Sim12-dado-exemplo2'

with open(nome, 'rb') as dbfile:
    dado = pickle.load(dbfile)
    dbfile.close()
    
t = dado['t']
rs = dado['rs']
r0 = dado['r0']
vs = dado['vs']

apenasz = len(np.shape(rs))==2 #booleano indicando se temos apenas as componentes em Z (True) ou se temos todas as 3 componentes(False) 
if not apenasz:
    r0 = r0[:,2]
    rs = rs[:, :, 2]
    vs = vs[:, :, 2]

ax.plot(*np.where(np.logical_and(t>0.2, t<0.6),rs, np.full(np.shape(rs), np.nan)), color='k', linewidth=0.1)
ax.plot(*r0, '.')
    
    
    

Utint = RectBivariateSpline(z0, z1, Ut)

Ut = Utint.ev(*rs)


#K = (m[0,0]/2)*((vs[0,:]+vs[1,:])**2+(vs[1,:]-vs[0,:])**2)
K = (m[0,0]/2)*((vs[0,:])**2+(vs[1,:])**2)

Et = np.squeeze( K+Ut)

fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
ax.plot(t, Ut,'-')

ax.plot(t, Et, '-')
ax.plot(t, K,'-')
ax.set_xlim(0.2, 0.6)

print((np.max(Et)-np.min(Et))/np.mean(Et))

# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# ax.plot(t, rs[0,:],'-')
# ax.plot(t, rs[1,:],'-')

# ax.set_xlim(0.2, 0.6)


# d =np.linspace(1.5, 7, 200)
# R = np.linspace(1.5, 7, 190)


 

# dl = np.expand_dims(d, 0)
# Rl = np.expand_dims(R, 1)



# filtro =np.logical_and( np.logical_and(np.less_equal((Rl-dl)/2, max(z0)), np.greater_equal((Rl-dl)/2, min(z0))), np.logical_and(np.less_equal((Rl+dl)/2, max(z1)), np.greater_equal((Rl+dl)/2, min(z1))))

# Utrd = np.where(filtro, Utint.ev((Rl-dl)/2,(dl+Rl)/2), np.full(np.shape(dl), np.nan))

# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(d, R, Utrd, levels=20)
# ax.set_title("Potencial acústico total do sistema")
# ax.set_xlabel(r'$d$ [mm]')
# ax.set_ylabel(r'$R$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$U_t$ []', format="{x:.1e}")
