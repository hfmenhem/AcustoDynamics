import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

f=40e3 #Hz
dicMeio = Simulador.ar(1)

Npar=2
a = np.array(Npar*[[1]]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
cPol = 2740*(10**3) #[mm/s] 
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume
f1 = 1- ((dicMeio['rho']*(dicMeio['c']**2))/ (rhoPol*(cPol**2)))
f2 = 2*((rhoPol-dicMeio['rho'])/((2*rhoPol)+dicMeio['rho']))
v0t = 1e3 #mm/s
Lamb=dicMeio["c"]/f

fase=0

print(f'lambda = {Lamb:.2f} mm ')

sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, 0, 1e3*dicMeio['dinvis'])
sim.tinyLev(fase)

g=[0,0,-9.81e3]

z0eq=[-Lamb/4, Lamb/4]
dz = 1
tsim = 1

#Achar ponto de quilíbrio em 2 partículas
dt = 1e-6
tcar = m[0,0]/(dicMeio['dinvis']*1e3)#tempo característico devido à força de arrasto

r0 = np.array([[[0.0,0,z0eq[0]]],[[0.0,0,z0eq[1]]]])
v0 = np.array([[[0.0,0,0]],[[0.0,0,0]]])

rs, vs, t = sim.Simular(r0, v0, dt, tcar, g=g)

while not np.all(np.isclose(vs[:,-1, :],0)): #Simula até que a velocidade seja nula
    rs, vs, t = sim.Simular(rs[:,-1,:], vs[:,-1,:], dt, tcar, g=g)
    
req = rs[:,-1,:] #pontos de equilíbrio

r0 = rs[:,-1,:]+np.array([[0,0,-dz],[0,0,-dz/2]])
#Religamento
sim = Simulador(np.array(Npar*[[f1]]), np.array(Npar*[[f2]]), f, dicMeio['c'], a, m, dicMeio['rho'], v0t, 0, 0*dicMeio['dinvis'])
sim.tinyLev(fase)
rs, vs, t = sim.Simular(r0, vs[:,-1,:], dt, tsim, g=g)


print('nodos de pressão, se fossem ondas planas')
print(f'z0 = {z0eq[0]:.2f} mm')
print(f'z1 = {z0eq[1]:.2f} mm')

print('pontos de equilíbrio, partículas simultâneas')
print(f'z0 = {req[0,2]:.2f} mm')
print(f'z1 = {req[1,2]:.2f} mm')

print('pontos após descer')
print(f'z0 = {r0[0,2]:.2f} mm')
print(f'z1 = {r0[1,2]:.2f} mm')

arraysalva=[['Amplitude de velocidade do transdutor', f'{v0t:.4e} mm/s'],
              ['deslocamento em z', f'{dz:.1f} mm'],
              ['Partículas de PP no ar', 'usando TinyLev'],
              ['nodos de pressão, se fossem ondas planas',''],
              ['z0' , f'{z0eq[0]:.4f} mm'], 
              ['z1' ,f'{z0eq[1]:.4f} mm'],
              ['pontos de equilíbrio, partículas simultâneas', ''],
              ['z0', f'{req[0,2]:.4f} mm'],
              ['z1', f'{req[1,2]:.4f} mm'],
              ['pontos após descer', ''],
              ['z0', f'{r0[0,2]:.4f} mm'],
              ['z1', f'{r0[1,2]:.4f} mm']
              
              ]

np.savetxt(f'Batimento-queda {dz:.0f}-posicao{z0eq[0]*2/Lamb:.1f};{z0eq[1]*2/Lamb:.1f}.txt', arraysalva, fmt='%s')

#x x t
plt.figure(dpi=300)
for i in range(Npar):
    plt.plot(t, rs[i, :, 2], linestyle='-', marker='',markersize=2)
#plt.plot(t, (rs[0, :, 2]+rs[1, :, 2])/2, linestyle='-', marker='',markersize=2)
    
plt.ylabel("z [mm]")
plt.xlabel("t [s]")
plt.title(f'Batimento - deslocamento em z {dz:.1f} mm \n Amplitude de velocidade do transdutor {v0t*1e-3:.1f} m/s')
plt.grid()

#plt.savefig(f'Batimento-queda {dz:.0f}-posicao{z0eq[0]*2/Lamb:.1f};{z0eq[1]*2/Lamb:.1f}.png')
plt.show()

plt.figure(dpi=300)
plt.plot( rs[0, :, 2], rs[1, :, 2], linestyle='-', marker='',linewidth =0.5)
plt.ylabel("z2 [mm]")
plt.xlabel("z1 [mm]")
plt.title(f'Espaço de fase - deslocamento em z {dz:.1f} mm \n Amplitude de velocidade do transdutor {v0t*1e-3:.1f} m/s')
plt.grid()
plt.show()

plt.figure(dpi=300)
plt.plot( rs[0, :, 2],vs[0, :, 2], linestyle='-', marker='',linewidth =0.5)
plt.ylabel("v1 [mm/s]")
plt.xlabel("z1 [mm]")
plt.title(f'Espaço de fase 2 - deslocamento em z {dz:.1f} mm \n Amplitude de velocidade do transdutor {v0t*1e-3:.1f} m/s')
plt.grid()
plt.show()