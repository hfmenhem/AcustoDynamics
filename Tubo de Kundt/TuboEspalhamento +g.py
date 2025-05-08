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
pressao = 800#Pa = g/mm*s^2
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
rs = np.array([[[0,0,0]]]) #Posições das partículas emissoreas

print(f'Z max = {Lamb} mm ')

Yrange = [0, Lamb/20]
Zrange = [-Lamb/20, Lamb/20]

Yrange = [0, 20]
Zrange = [-20, 20]

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
rf = [0,Lamb,0]
d=np.linalg.norm(rp - rf, axis=2)[:,0]
arg = np.argsort(d)
U0 = U.flatten()[arg][0]


#Plotar o potencial
#filtro:
Filtro = np.linalg.norm(rp, axis=2)>a[0,0]
nan = np.empty((200,400))
nan[:,:]= np.nan

UrF = np.where(np.reshape(Filtro, (200,400)),U, nan)

fig3, ax3 = plt.subplots(dpi=300, figsize=(10,10))
ax3.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
#pcm3 = ax3.pcolormesh(ys, zs, (10**9)*np.transpose(UrF), norm=colors.SymLogNorm(linthresh=10**7))
pcm3 = ax3.contourf(ys, zs, (10**9)*np.transpose(UrF), levels=40)

fig3.colorbar(pcm3, ax=ax3, label='Usc [aJ]')
ax3.set_title('Potencial de Gor\'kov de espalhamento')
ax3.set_xlabel('y [mm]')
ax3.set_ylabel('z [mm]')
fig3.show()



#Delta Potencial a partir do elemento da extremidade -y,-z
FY4 = F.reshape(len(ys),(len(zs)), 3)[:,:,1]
FZ4 = F.reshape(len(ys),(len(zs)), 3)[:,:,2]

#Lista de integrais em y andando dodos os passos
FTyintU= np.repeat([FY4[:,0]], len(ys), axis=0)
#Filtro: matriz len(ys) X len(ys)
Ffiltro=np.ones((len(ys), len(ys)))
Ffiltro=np.tril(Ffiltro)
Ffiltro = np.transpose(Ffiltro)
FTyintU=np.sum(FTyintU*Ffiltro, axis = 1)*dy

#Lista de integrais em z andando todos os passos
FTzintU= np.repeat([FZ4], len(zs), axis=0)
#Filtro: matriz len(zs) X len(zs) e depois estendida para a dimnensão de len(ys)
Ffiltro=np.ones((len(zs), len(zs)))
Ffiltro=np.tril(Ffiltro)
Ffiltro = np.transpose(Ffiltro)
Ffiltro = np.repeat(np.expand_dims(Ffiltro, 1), len(ys), axis=1)
FTzintU=np.transpose(np.sum(FTzintU*Ffiltro, axis = 2)*dz)

UT = FTzintU + np.expand_dims(FTyintU, 1)

#Trocar a referência de 0 do potencial
# rf = [0,Lamb,0]
# d=np.linalg.norm(rp - rf, axis=2)[:,0]
# arg = np.argsort(d)
# UT0 = UT.flatten()[arg][0]
UT0 = np.min(UT)


#Plotar o potencial
#filtro:
Filtro = np.linalg.norm(rp, axis=2)>a[0,0]
nan = np.empty((200,400))
nan[:,:]= np.nan

UTrF = np.where(np.reshape(Filtro, (200,400)),UT, nan)

fig4, ax4 = plt.subplots(dpi=300, figsize=(10,10))
ax4.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
#pcm4 = ax4.pcolormesh(ys, zs, (10**9)*np.transpose(UTrF), norm=colors.SymLogNorm(linthresh=10**7))
pcm4 = ax4.contourf(ys, zs, (10**9)*np.transpose(UTrF),levels=40)

fig4.colorbar(pcm4, ax=ax4, label='Ut [aJ]')
ax4.set_title('Potencial de Gor\'kov total')
ax4.set_xlabel('y [mm]')
ax4.set_ylabel('z [mm]')
fig4.show()


# figH, axH = plt.subplots(dpi=300)
# UrFhist = UrF.flatten()
# axH.hist(UrFhist, bins=int(len(UrFhist)/100), log=True)
# figH.show()

fig5, ax5 = plt.subplots(dpi=300, figsize=(10,10))
ax5.set_aspect(1)

Ug =-1* m[0,0]*np.einsum('ijk, k -> ij', rp, g).reshape((200,400))

pcm5 = ax5.contourf(ys, zs, (10**9)*np.transpose(UTrF+Ug), levels = 20)
#pcm5 = ax5.pcolormesh(ys, zs, (10**9)*np.transpose(Ug))

fig5.colorbar(pcm5, ax=ax5, label='U [aJ]')
ax5.set_title('Potencial de Gor\'kov total + Potencial gravitacional')
ax5.set_xlabel('y [mm]')
ax5.set_ylabel('z [mm]')
fig5.show()
