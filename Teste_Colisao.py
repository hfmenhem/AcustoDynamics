from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

cmap = mpl.colormaps['viridis']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


Npar = 2


# dicMeio = Simulador.agua(Npar)
# a = np.array(Npar*[[0.1*np.pi/dicMeio['k']]]) # [mm]
# m = np.array([[0.01], [0.01]]) # [g]
#h=np.pi/dicMeio['k']

dicMeio = Simulador.ar(Npar)
teste=dicMeio['k']
a = np.array(Npar*[[0.1*np.pi/dicMeio['k']]]) # [mm]
m = (a**3*(4*np.pi/3))*(dicMeio['rho']*1000) # [g], densidade do ar vezes seu volume

h=0

teste=dicMeio['f1']
teste2=np.array([dicMeio['f1'][0]])

#Perfil da força de Gor'kov
sim = Simulador(np.array([dicMeio['f1'][0]]), np.array([dicMeio['f2'][0]]), dicMeio['f'], dicMeio['c'], np.array([a[0]]), np.array([m[0]]), dicMeio['rho'], dicMeio['v0']/10, h)
zs = np.linspace(-1*np.pi/dicMeio['k'], np.pi/dicMeio['k'], 300)
r = np.expand_dims([0,0,1]*np.expand_dims(zs, 1),1)

Pin = sim.PhiIn(r) #[mm^2/s]
GPin = sim.GradPhiIn(r)
HPin = sim.HPhiIn(r)
F = sim.FGorKov(Pin, GPin, HPin) #[uN]

plt.figure(dpi=300)
plt.plot(r[:, 0, 2],F[:, 0, 2], linestyle='', marker='.',markersize=2,  label='F programa')
plt.title("Perfil da força de Gor'kov")
plt.xlabel("z [mm]")
plt.ylabel("Força [g*mm/s^2 = nN]")
plt.legend()

#-----------

# r0 = np.array([[[-0.25*np.pi/dicMeio['k'],0,0]], [[+0.25*np.pi/dicMeio['k'],0,0]]])
# v0 = np.array([[[0.02,0,0.002]], [[-0.02,0,0]]])

r0 = np.array([[[-0.1*np.pi/dicMeio['k'],0,0]], [[+0.1*np.pi/dicMeio['k'],0,0]]])
v0 = np.array([[[0.0,0,0]], [[-0.0,0,0]]])


sim = Simulador(dicMeio['f1'], dicMeio['f2'], dicMeio['f'], dicMeio['c'], a, m, dicMeio['rho'], dicMeio['v0']/10, h)
dt = 0.001
tempo = 10

rs, vs, t, TColsisoes = sim.SimularComColisão(r0, v0, dt, tempo)
CorTempo = cmap(np.linspace(0, 1, len(t))) #Mapa de cor para indicar o tempo da simulação

#Achar tempos para plotar circulos
ti = np.linspace(0, np.max(t), 5)
indTi = np.argmin((np.abs(t - np.expand_dims(ti, 1))),axis= 1)

#Achando indices da colisão
indCol = np.any(np.equal(np.expand_dims(t, 1), np.expand_dims(TColsisoes,0)), axis=1)
indtempocol = np.arange(0, len(indCol),1)[indCol]
rColisao  = rs[:, indCol,:]

#achando todos os pontos em que queremos plotar as esferas como circulos
indplotR = np.concatenate((indTi, indtempocol))
rplotR = rs[:, indplotR, :]


#Plot do plano xz
plt.figure(dpi=300)
plt.axes().set_aspect('equal')
#Definição do circulo unitário
ang=np.linspace(0, 2*np.pi, 50)
circ = np.transpose([np.sin(ang), np.zeros(len(ang)), np.cos(ang)])

for i in range(Npar):
    plt.plot(rs[i, :, 0], rs[i, :, 2], linestyle='', marker='.',markersize=2)
    plt.plot(rColisao[i, :, 0], rColisao[i, :, 2], linestyle='', marker='.',markersize=2)
    
    cirplotR=rplotR[i, :,:] + np.expand_dims(a[i]*circ, 1)
    for c, indc in enumerate(indplotR):
        plt.plot(cirplotR[:,c, 0], cirplotR[:, c,2], linestyle='', marker='.',markersize=1, color=CorTempo[indc])


plt.xlabel("x [mm]")
plt.ylabel("z [mm]")

plt.show()

#x x t
plt.figure(dpi=300)
for i in range(Npar):
    plt.plot(t, rs[i, :, 0], linestyle='', marker='.',markersize=2)
    plt.plot(TColsisoes, rColisao[i, :, 0], linestyle='', marker='.',markersize=2)

plt.ylabel("x [mm]")
plt.xlabel("t [s]")

plt.show()

#z x t
plt.figure(dpi=300)
for i in range(Npar):
    plt.plot(t, rs[i, :, 2], linestyle='', marker='.',markersize=2)
    plt.plot(TColsisoes, rColisao[i, :, 2], linestyle='', marker='.',markersize=2)

plt.ylabel("z [mm]")
plt.xlabel("t [s]")

plt.show()

#y x t
plt.figure(dpi=300)
for i in range(Npar):
    plt.plot(t, rs[i, :, 1], linestyle='', marker='.',markersize=2)
    plt.plot(TColsisoes, rColisao[i, :, 1], linestyle='', marker='.',markersize=2)

plt.ylabel("y [mm]")
plt.xlabel("t [s]")

plt.show()