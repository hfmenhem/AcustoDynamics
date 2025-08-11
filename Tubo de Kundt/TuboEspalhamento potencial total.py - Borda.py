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
pressao =680#Pa = g/mm*s^2
#pressao = 2000#Pa = g/mm*s^2

a = np.array([[3]]) # [mm]
rhoPol = (20*(10**-6))
cPol = 2350*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

g =np.array( [0,-9.81e3, 0]) * (1-(dicMeio['rho']/rhoPol))#g efetivo considerando empuxo

Lamb=dicMeio["c"]/f

print(r'$\lambda = $'+f'{Lamb:.2e}mm')


#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
#Vo = P0*k/omega*Rho
v0 = pressao/(dicMeio['rho']*dicMeio['c'])

h=0

sim = Simulador(np.array([[f1]]), np.array([[f2]]),f, dicMeio['c'], a, m, dicMeio['rho'], v0, h, 0)

thetas = np.linspace(0, np.pi, 400)
xas = np.linspace(-40,40, 200)

UTs =[]
for xa in xas:
    xb = 2*a[0,0]*np.cos(thetas) + xa
    zb = 2*a[0,0]*np.sin(thetas)
    
    rs = np.array([[[0,0,xa]]]) #Posição da partículas emissora
    
    rp = np.expand_dims(np.transpose([np.zeros(len(zb)), zb, xb]), 1) 
    
    MR = rp-rs
    PinB = sim.PhiIn(rp) #[mm^2/s]
    GPinB = sim.GradPhiIn(rp)

    PinA = sim.PhiIn(rs) #[mm^2/s]
    GPinA = sim.GradPhiIn(rs)

    PscB = sim.PhiSc(MR, np.transpose(PinA), np.transpose(GPinA, axes=(1,0,2)))
    GPscB = sim.GradPhiSc(MR, np.transpose(PinA), np.transpose(GPinA, axes=(1,0,2)))

    PscA = sim.PhiSc(np.transpose(MR, axes = (1,0,2)), np.transpose(PinB), np.transpose(GPinB, axes=(1,0,2)))
    GPscA = sim.GradPhiSc(np.transpose(MR, axes = (1,0,2)), np.transpose(PinB), np.transpose(GPinB, axes=(1,0,2)))

    PtB  = PscB  + PinB
    GPtB = GPscB + GPinB

    PtA  = PscA  + PinA
    GPtA = GPscA + GPinA

    UB = sim.UGorkov(PtB, GPtB)
    UA = np.transpose(sim.UGorkov(PtA, GPtA))
    UT = (UA+UB)
    

    Ug =-1* m[0,0]*np.einsum('ijk, k -> ij', rp, g)
    
    UT = UT + Ug
    UTs.append(np.squeeze(UT))
    
UTs = np.array(UTs)

fig, ax = plt.subplots(dpi=300, figsize=(10,10))
#ax.set_aspect(1)
pcm = ax.contourf(thetas, xas, UTs,levels=40)

custom_ticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
custom_tick_labels = ['$0$', '$\pi/4$', '$\pi/2$','$3\pi/4$', '$\pi$']

ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_tick_labels)


fig.colorbar(pcm, ax=ax, label='Ut [J]')
ax.set_title(r'Potencial total na fronteira $|\vec d | = 2a$')
ax.set_xlabel(r'$\theta$ [rad]')
ax.set_ylabel('$x_A$ [mm]')
fig.show()

