import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import RectBivariateSpline
#Este código testa se o programa calcula corretamente a força acústica secundária


#f=240 #Hz
dicMeio = Simulador.ar(1)

pressoes = np.linspace(500, 800, 30)
pressoes = np.linspace(600-1, 600+5, 30)
fs = np.linspace(100, 1000, 40)
#pressao = 500#Pa = g/mm*s^2


#pressao = 2000#Pa = g/mm*s^2

a = np.array([[3]]) # [mm]
rhoPol = (20*(10**-6))
cPol = 2350*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume

f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))

g =np.array( [0,-9.81e3, 0]) * (1-(dicMeio['rho']/rhoPol))#g efetivo considerando empuxo

#Lamb=dicMeio["c"]/f

#print(r'$\lambda = $'+f'{Lamb:.2e}mm')


#v0 = (np.sqrt(2)/10) * (10**3) # [mm/s] velocidade equivalente para uma energia de 10J
#Vo = P0*k/omega*Rho
#v0 = pressao/(dicMeio['rho']*dicMeio['c'])
v0s = pressoes/(dicMeio['rho']*dicMeio['c'])

h=0


dtheta = np.pi/10
dxa = 1
thetas = np.linspace((np.pi/2) - dtheta, (np.pi/2) + dtheta, 40)
xas = np.linspace(-dxa,dxa, 10)

grad02=[]
H2 =[]
eminimo2 =[]

for f in fs:
    grad0=[]
    H =[]
    eminimo =[]
    for v0 in v0s:
        sim = Simulador(np.array([[f1]]), np.array([[f2]]),f, dicMeio['c'], a, m, dicMeio['rho'], v0, h, 0)
        
        
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
        UTint = RectBivariateSpline(xas,thetas, UTs)
        grad0.append(np.isclose(UTint.ev(0, np.pi/2, 1,0), 0) and np.isclose(UTint.ev(0, np.pi/2, 0, 1), 0) )
        H.append((UTint.ev(0, np.pi/2, 2,0)*UTint.ev(0, np.pi/2, 0,2))-(UTint.ev(0, np.pi/2, 1,1)**2))
        eminimo.append( UTint.ev(0, np.pi/2, 2, 0)>0)

    grad02.append(grad0)
    H2.append(H)
    eminimo2.append(eminimo)
    
grad02 = np.array(grad02)
H2 = np.array(H2)
eminimo = np.array(eminimo2)


fig, ax = plt.subplots(dpi=300)
#plt.plot(pressoes, H, '.')
#ax.set_aspect(1)
pcm = ax.contourf(fs, pressoes,np.transpose( H2), levels=20)
pcm2 = ax.contour(fs, pressoes,np.transpose( H2), colors = 'black', levels=[0])

# fig.colorbar(pcm, ax=ax, label=r'Hessiana de $U_T$ [J/mm^2]')
# ax.set_title(r'Hess($U_T$) para determinação de estabilidade')
# ax.set_xlabel(r'frequência [Hz]')
# ax.set_ylabel('Pressão [Pa]')
# fig.show()


ax.minorticks_on()

fig.colorbar(pcm, ax=ax, label=r'Hessiana de $U_T$ [J/mm^2]')
ax.set_title(r'Recorte - região onde Hess($U_T$)$\approx 0$')
ax.set_xlabel(r'frequência [Hz]')
ax.set_ylabel('Pressão [Pa]')
fig.show()

