import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft, fftfreq

f=40e3 #Hz
dicMeio = Simulador.ar(1)

Npar=2
a = np.array(Npar*[[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume
f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))
v0t = 10e3 #mm/s
Lamb=dicMeio["c"]/f

h=0

print(f'lambda = {Lamb:.2f} mm ')

sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, h, 1e3*dicMeio['dinvis'])


g=[0,0,-9.81e3]

z0eq=[0, Lamb/2]
#dzeq=[-.05, 0.0]
dzeq=[-1.5, 0.0]

tsim = 1

#=====================Achar ponto de quilíbrio em 2 partículas=====================
dt = 1e-3
tcar = m[0,0]/(dicMeio['dinvis']*1e3)#tempo característico devido à força de arrasto

r0 = np.array([[[0.0,0,z0eq[0]]],[[0.0,0,z0eq[1]]]])
v0 = np.array([[[0.0,0,0]],[[0.0,0,0]]])

rs, vs, t = sim.Simular(r0, v0, dt, tcar, g=g)

while not np.all(np.isclose(vs[:,-1, :],0)): #Simula até que a velocidade seja nula
    rs, vs, t = sim.Simular(rs[:,-1,:], vs[:,-1,:], dt, tcar, g=g)
    print(vs[:,-1,:])
    
req = rs[:,-1,:] #pontos de equilíbrio

r0 = rs[:,-1,:]+np.array([[0,0,dzeq[0]],[0,0,dzeq[1]]])


#=====================Religamento=====================
sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, h, 0*dicMeio['dinvis'])
rs, vs, t = sim.Simular(r0, vs[:,-1,:], dt, tsim, g=g)


print('nodos de pressão, se fossem ondas planas')
print(f'z0 = {z0eq[0]:.2f} mm')
print(f'z1 = {z0eq[1]:.2f} mm')

print('pontos de equilíbrio, partículas simultâneas')
print(f'z0 = {req[0,2]:.2f} mm')
print(f'z1 = {req[1,2]:.2f} mm')

print('deslocamento incial em z')
print(f'z0 = {dzeq[0]:.2f} mm')
print(f'z1 = {dzeq[1]:.2f} mm')

plt.figure(dpi=300)
fig, axs = plt.subplots(Npar, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

#x x t

for i in range(Npar):
    axs[i].plot(t, rs[i, :, 2], linestyle='-', marker='',markersize=2)    
    axs[i].grid()
#plt.plot(t, (rs[0, :, 2]+rs[1, :, 2])/2, linestyle='-', marker='',markersize=2)
    
# axs[0].spines.top.set_visible(True)
# axs[-1].spines.bottom.set_visible(True)
# axs[-1].xaxis.tick_bottom()
fig.supylabel("z [mm]")
fig.supxlabel("t [s]")
fig.suptitle(f'Batimento - deslocamento em z ({dzeq[0]:.1f},{dzeq[1]:.1f}) mm \n Amplitude de velocidade do transdutor {v0t*1e-3:.1f} m/s')
#plt.savefig(f'Batimento-queda {dz:.0f}-posicao{z0eq[0]*2/Lamb:.1f};{z0eq[1]*2/Lamb:.1f}.png')
plt.show()


plt.figure(dpi=300)
fig, axs = plt.subplots(Npar, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

yfs=[]
xfs=[]
flim=1e-3
for i in range(Npar):
    N=len(rs[i, :, 2])
    yf=fft(rs[i, :, 2])
    xf=fftfreq(N, dt)[:N//2]
    yfs.append(yf)
    xfs.append(xf)
    
    yplot= 2.0/N * np.abs(yf[0:N//2]) 
    filtro =yplot>flim
    nans=np.full(np.shape(yplot), np.nan)
    
    axs[i].plot(np.where(filtro,xf, nans)[1:], np.where(filtro,yplot, nans)[1:], ',')
    #axs[i].grid()
    
fig.supylabel("amplitude [mm]")
fig.supxlabel("frequência [Hz]")
fig.suptitle(f'Transformada de Fourier - deslocamento em z ({dzeq[0]:.1f},{dzeq[1]:.1f}) mm \n Amplitude de velocidade do transdutor {v0t*1e-3:.1f} m/s')

np.save('rs2', rs)
np.save('vs2', vs)
np.save('t2', t)

plt.show()


