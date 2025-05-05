from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#Este código testa se o programa calcula corretamente a força acústica secundária


# Dados de (SILVA, BRUUS, 2014)
# f1 = np.array([[-0.08]])
# f2 = np.array([[0.07]])
# f = 2*(10**6) #[1/s]
# c = 1500*(10**3) #[mm/s]
# k = 2*np.pi*f/c #[1/mm]
# a = np.array([[0.1/k]]) # [mm]
# m = (a**3*(4*np.pi/3))*(1*10**3) # [g], massa, não importa
# rho = 1000*(10**(-6)) #g/mm^3
# v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J

#Dados de (SIMON et al., 2019)
f1 = np.array([[0.623]])
f2 = np.array([[0.034]])
f = 10*(10**6) #[1/s]
c = 1480*(10**3) #[mm/s]
k = 2*np.pi*f/c #[1/mm]
a = np.array([[0.1*np.pi/k]]) # [mm]
m = (a**3*(4*np.pi/3))*(1*10**3) # [g], massa, não importa
rho = 998*(10**(-6)) #g/mm^3
#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
#Vo = P0*k/omega*Rho
v0 = (50*(10**3))*k/(2*np.pi*f*rho) #Pressão = 50*(10**3) Pa = 50*(10**3) g/mm*s^2

h=(np.pi/2)/k

sim = Simulador(f1, f2, f, c, a, m, rho, v0, h, 0)

rs = np.array([[[0,0,0]]]) #Posições das partículas emissoreas

print(f'Z max = {2*np.pi/k} mm ')

Yrange = [0, 2*np.pi/k]
Zrange = [-2*np.pi/k, 2*np.pi/k]

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

Pt  = np.sum(Psc , axis = 1,keepdims=True) + Pin
GPt = np.sum(GPsc, axis = 1,keepdims=True) + GPin
HPt = np.sum(HPsc, axis = 1,keepdims=True) + HPin

# Pt  = Pin
# GPt = GPin
# HPt = HPin

F = sim.FGorKov(Pt, GPt, HPt) #[uN]
Fin = sim.FGorKov(Pin, GPin, HPin)
Fsc = F-Fin

Fscn = np.linalg.norm(Fsc, axis=2)[:,0]
Fscn= Fscn.reshape(len(ys),(len(zs)))

#filtro:
Filtro = Fscn<3*(10**(-7))
nan = np.empty(np.shape(Fscn))
nan[:,:]= np.nan
FscnF = np.where(Filtro,Fscn, nan)

#Plotar módulo da força
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
ax.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
pcm = ax.pcolormesh(ys, zs, np.transpose(FscnF))
ax.set_title("Módulo da força acústica")
ax.set_xlabel('y [mm]')
ax.set_ylabel('z [mm]')
fig.colorbar(pcm, ax=ax, label='|Fsc| [uN]')

fig.show()


#Integral de linha fechada da força
passos = 10

ys2 = np.concatenate((ys, np.linspace(Yrange[1]+dy, Yrange[1]+(dy*(passos-1)), (passos-1))))
zs2 = np.concatenate((zs, np.linspace(Zrange[1]+dz, Zrange[1]+(dz*(passos-1)), (passos-1))))

z2 = np.expand_dims([0,0,1]*np.expand_dims(zs2, 1),0)
y2 = np.expand_dims([0,1,0]*np.expand_dims(ys2, 1),1)

rp2 = (z2+y2).reshape(-1,1,3) #Posições das partículas de prova

MR2 = rp2-rs
Pin2 = sim.PhiIn(rp2) #[mm^2/s]
GPin2 = sim.GradPhiIn(rp2)
HPin2 = sim.HPhiIn(rp2)

Psc2 = sim.PhiSc(MR2, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)) )
GPsc2 = sim.GradPhiSc(MR2, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))
HPsc2 = sim.HPhiSc(MR2, np.transpose(sim.PhiIn(rs)), np.transpose(sim.GradPhiIn(rs), axes=(1,0,2)))

Pt2  = np.sum(Psc2 , axis = 1,keepdims=True) + Pin2
GPt2 = np.sum(GPsc2, axis = 1,keepdims=True) + GPin2
HPt2 = np.sum(HPsc2, axis = 1,keepdims=True) + HPin2

F2 = sim.FGorKov(Pt2, GPt2, HPt2) #[uN]
Fin2 = sim.FGorKov(Pin2, GPin2, HPin2)
Fsc2 = F2-Fin2

FscY2 = Fsc2.reshape(len(ys2),(len(zs2)), 3)[:,:,1]
FscZ2 = Fsc2.reshape(len(ys2),(len(zs2)), 3)[:,:,2]


#Lista de integrais em y andando N passos
Fyint= np.repeat([FscY2], len(ys2)-passos+1, axis=0)
#Filtro: matriz len(ys) X len(len(ys)-Passos) e depois estendida para a dimnensão de len(zs)
Ffiltro=np.zeros((len(ys2), len(ys2)))
for i in range(passos):
    Ffiltro= Ffiltro+np.eye(len(ys2), k=-i)
Ffiltro = np.transpose(Ffiltro[:,:-passos+1])
Ffiltro = np.repeat(np.expand_dims(Ffiltro, 2), len(zs2), axis=2)
Fyint=np.sum(Fyint*Ffiltro, axis = 1)*dy

#Lista de integrais em z andando N passos
Fzint= np.repeat([FscZ2], len(zs2)-passos+1, axis=0)
#Filtro: matriz len(zs) X len(len(zs)-Passos+1) e depois estendida para a dimnensão de len(ys)
Ffiltro=np.zeros((len(zs2), len(zs2)))
for i in range(passos):
    Ffiltro= Ffiltro+np.eye(len(zs2), k=-i)
Ffiltro = np.transpose(Ffiltro[:,:-passos+1])
Ffiltro = np.repeat(np.expand_dims(Ffiltro, 1), len(ys2), axis=1)
Fzint=np.transpose(np.sum(Fzint*Ffiltro, axis = 2)*dz)

IntO = Fzint[:-passos+1,:]-Fzint[passos-1:,:] -( Fyint[:,:-passos+1]-Fyint[:,passos-1:])

#Plotar Integral de linha fechada da força (Checar se é potencial)

fig2, ax2 = plt.subplots(dpi=300, figsize=(10,10))
ax2.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
pcm2 = ax2.pcolormesh(ys, zs, (10**9)*np.abs(np.transpose(IntO)),norm=colors.LogNorm(vmax=(10**9)*IntO.max()))

fig2.colorbar(pcm2, ax=ax2, label='Módulo da integral fechada de Fsc [aJ]')
ax2.set_title(f'Módulo da Integral fechada de Fsc no retângulo de lados \n Δy = {passos*(Yrange[1]-Yrange[0])/(len(ys)-1):.2E} mm e Δz = {passos*(Zrange[1]-Zrange[0])/(len(zs)-1):.2E} mm ')
ax2.set_xlabel('y [mm]')
ax2.set_ylabel('z [mm]')
fig2.show()

#Delta Potencial a partir do elemento da estremidade -y,-z
FscY3 = Fsc.reshape(len(ys),(len(zs)), 3)[:,:,1]
FscZ3 = Fsc.reshape(len(ys),(len(zs)), 3)[:,:,2]

#Lista de integrais em y andando dodos os passos
FyintU= np.repeat([FscY3[:,0]], len(ys), axis=0)
#Filtro: matriz len(ys) X len(ys)
Ffiltro=np.ones((len(ys), len(ys)))
Ffiltro=np.tril(Ffiltro)
Ffiltro = np.transpose(Ffiltro)
FyintU=np.sum(FyintU*Ffiltro, axis = 1)*dy

#Lista de integrais em z andando todos os passos
FzintU= np.repeat([FscZ3], len(zs), axis=0)
#Filtro: matriz len(zs) X len(zs) e depois estendida para a dimnensão de len(ys)
Ffiltro=np.ones((len(zs), len(zs)))
Ffiltro=np.tril(Ffiltro)
Ffiltro = np.transpose(Ffiltro)
Ffiltro = np.repeat(np.expand_dims(Ffiltro, 1), len(ys), axis=1)
FzintU=np.transpose(np.sum(FzintU*Ffiltro, axis = 2)*dz)

U = FzintU + np.expand_dims(FyintU, 1)

#Trocar a referência de 0 do potencial
rf = [0,2*np.pi/k,0]
d=np.linalg.norm(rp - rf, axis=2)[:,0]
arg = np.argsort(d)
U0 = U.flatten()[arg][0]

#Plotar o potencial
#filtro:
Filtro = np.abs(U-U0)<5*(10**(-9))
nan = np.empty(np.shape(U))
nan[:,:]= np.nan
UrF = np.where(Filtro,U-U0, nan)

fig3, ax3 = plt.subplots(dpi=300, figsize=(10,10))
ax3.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
pcm3 = ax3.pcolormesh(ys, zs, (10**9)*np.transpose(UrF))

fig3.colorbar(pcm3, ax=ax3, label='Usc [aJ]')
ax3.set_title(f'Potencial de Gor\'kov com Uo = U({rf[0]:.3f}, {rf[1]:.3f}, {rf[2]:.3f} mm)')
ax3.set_xlabel('y [mm]')
ax3.set_ylabel('z [mm]')
fig3.show()



