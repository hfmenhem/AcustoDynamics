import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior


from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

#Este código testa se o programa calcula corretamente a força acústica secundária



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

v0 = pressao/(dicMeio['rho']*dicMeio['c'])

Lamb=dicMeio["c"]/f

h=0

sim = Simulador(np.array([[f1]]), np.array([[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0, h, 0)

print(f'lambda = {Lamb:.2f} mm ')

#Yrange = [0, 2*np.pi/k]
#Dz = [Lamb*(1+(1/8)), Lamb*(1-(1/8))]
#Zranges = [-Lamb, Lamb]

Dz = [Lamb*(1-(1/8)), Lamb*(1+(1/8))]
Zranges = [Lamb*(-1), Lamb*(1)]

#Zrangep = [-Lamb, Lamb]
Zrangep = [Zranges[0]+Dz[0], Zranges[1]+Dz[1]]


#ys, dy = np.linspace(*Yrange, 200,retstep=True)
zs, dzs = np.linspace(*Zranges, 400,retstep=True)
Dzp = np.linspace(*Dz, 200)
zp = np.expand_dims(zs,0)+ np.expand_dims(Dzp, 1)

#rs = np.array([[[0,0,0]]]) #Posições das partículas emissoreas



Fscz=np.empty(np.shape(zp))
for i, zpi in enumerate(np.transpose(zp)):
    
    rp =np.array([[[0,0,1]]])*np.expand_dims(zpi, (1,2)) #posição das partículas de prova
    rs =np.array([[[0,0,1]]])*np.expand_dims([zs[i]], (0,2)) #posição das partículas espalhadoras
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
    
    # Pt  = Pin
    # GPt = GPin
    # HPt = HPin
    
    F = sim.FGorKov(Pt, GPt, HPt) #[uN]
    Fin = sim.FGorKov(Pin, GPin, HPin)
    Fsc = F-Fin
    
    Fscz[:, i] = Fsc[:,0,2]


#filtro:
Filtro = np.abs(Fscz)<1*(10**(9))
nan = np.full(np.shape(Fscz), np.nan)
FsczF = np.where(Filtro,Fscz, nan)


#Plotar  força no eixo z
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
#pcm = ax.contourf(zs, Dzp, FsczF, levels=20, locator=ticker.SymmetricalLogLocator(base=10 , linthresh=1e1), norm=colors.SymLogNorm(base=10 , linthresh=1e1,))
pcm = ax.contourf(zs, Dzp, FsczF, levels=20)

ax.set_title("força acústica secundária no eixo z" + f' - h = {h/Lamb:.2f}'+r'$\lambda$')
ax.set_xlabel(r'$z_s$ [mm]')
ax.set_ylabel(r'$\Delta z$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")

fig.savefig('teste1.png')
#fig.show()

#=================Plot zs X zp ===========================================

zp2 = np.linspace(*Zrangep, 400,retstep=False)

rp =np.array([[[0,0,1]]])*np.expand_dims(zp2, (1,2)) #posição das partículas de prova
rs =np.array([[[0,0,1]]])*np.expand_dims(zs, (0,2)) #posição das partículas espalhadoras
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
Fin = sim.FGorKov(Pin, GPin, HPin)
Fsc = F-Fin

Fscz2 = Fsc[:,:,2]

#filtro:
Filtro = np.abs(Fscz2)<1*(10**(9))
nan = np.full(np.shape(Fscz2), np.nan)
FsczF2 = np.where(Filtro,Fscz2, nan)

#filtro 2:
Filtro2 =np.logical_and(np.logical_and((rp-rs)[:,:,2]>=Dz[0], (rp-rs)[:,:,2]<=Dz[1]), np.logical_and(rs[:,:,2]>=Zranges[0], rs[:,:,2]<=Zranges[1]))
nan = np.full(np.shape(Fscz2), np.nan)
FsczF2 = np.where(Filtro2,Fscz2, nan)


#Plotar  força no eixo z
fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
#pcm = ax.pcolor(zs, zs, FscnF,norm=colors.LogNorm(vmax=FscnF.max()))
#pcm = ax.contourf(zs, zp2, FsczF2, levels=20, locator=ticker.SymmetricalLogLocator(base=10 , linthresh=1e1), norm=colors.SymLogNorm(base=10 , linthresh=1e1,))
pcm = ax.contourf(zs, zp2, FsczF2, levels=20)

ax.set_title("força acústica secundária no eixo z" + f' - h = {h/Lamb:.2f}'+r'$\lambda$')
ax.set_xlabel(r'$z_s$ [mm]')
ax.set_ylabel(r'$z_p$ [mm]')
fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")

fig.savefig('teste2.png')
#fig.show()


# def Fteste(D,z):
#     return (0.1*np.sin(2*np.pi*z/(Lamb)))+np.cos(np.pi*D/(Lamb))

# #Plotar  força de teste
# fig, ax = plt.subplots(dpi=300, figsize=(10,10))
# #ax.set_aspect(1)
# pcm = ax.contourf(zs, Dzp, Fteste(np.expand_dims(Dzp, 1), np.expand_dims(zs, 0)), levels=20)

# ax.set_title("força acústica secundária no eixo z" + f' - h = {h/Lamb:.2f}'+r'$\lambda$')
# ax.set_xlabel(r'$z_s$ [mm]')
# ax.set_ylabel(r'$z_p$ [mm]')
# fig.colorbar(pcm, ax=ax, label=r'$\hat z\cdot  Fsc z$ [uN]', format="{x:.1e}")

# fig.show()
 