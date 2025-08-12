import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import pickle
from scipy.optimize import root
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker



def plot(var0, var1, titulo, subt = None, label='', tipo = 1):

    fig, ax = plt.subplots(1,2,dpi=300)
    fig.suptitle(titulo)
    if subt is None:
        ax[0].set_title('partícula A')
        ax[1].set_title('partícula B')
    else:
        ax[0].set_title(subt[0])
        ax[1].set_title(subt[1])
        
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    
    t1=np.min(np.where(np.isnan([var0, var1]),np.inf, [var0, var1]))
    t2=np.max(np.where(np.isnan([var0, var1]),-np.inf, [var0, var1]))
    norm = mpl.colors.Normalize(vmin=t1, vmax=t2, clip=False)
    
    psm0 = ax[0].pcolormesh(zs[:,:,0],zs[:,:,1], var0,shading='nearest', norm=norm)
    psm1 = ax[1].pcolormesh(zs[:,:,0],zs[:,:,1], var1,shading='nearest', norm=norm)
    
    for i in range(2):
        if tipo == 1:
            ax[i].xaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].xaxis.set_minor_locator(ticker.LinearLocator(31))
            ax[i].yaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].yaxis.set_minor_locator(ticker.LinearLocator(31))
            
            ax[i].tick_params(axis="x", labelrotation=0)
            ax[i].set_xlabel('$z_o$ - partícula A [mm]')
        elif tipo == 2:
            ax[i].xaxis.set_major_locator(ticker.LinearLocator(2))
            ax[i].xaxis.set_minor_locator(ticker.LinearLocator(11))
            ax[i].yaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].yaxis.set_minor_locator(ticker.LinearLocator(31))
            
            ax[i].tick_params(axis="x", labelrotation=-10)
            ax[i].set_xlabel('$z_o$ - partícula A [mm]')
            
    
    if tipo == 1:
        ax[0].set_ylabel('$z_o$ - partícula B [mm]')
        ax[1].tick_params(axis="y", which = 'both', labelleft=False, right=True)
        ax[0].tick_params(axis="y", which = 'both', right=True)
        cbar = fig.colorbar(psm1, ax=ax, location='bottom', pad=0.21, label=label)
    elif tipo == 2:
        ax[0].set_ylabel('$z_o$ - partícula B [mm]')
        ax[1].tick_params(axis="y", which = 'both', labelleft=False, right=True)
        ax[0].tick_params(axis="y", which = 'both', right=True)
        cbar = fig.colorbar(psm1, ax=ax, location='right', label=label)
    
    return ax, cbar
    
#Organizando os dados
def organiza(arr):
    
    if  len(np.shape(r0s))==2:
        zs = r0s
    else:
        zs = r0s[:,:,2]

    arg = np.argsort(zs[:,0])
    zs = zs[arg, :]
    arr = arr[arg,:]

    N = int(np.sqrt(len(zs)))
    zs = np.reshape(zs, [N,N, 2])
    arr = np.reshape(arr, [N,N, 2])

    arg2 = np.argsort(zs[:,:,1], axis=1)
    arr = np.take_along_axis(arr, np.expand_dims(arg2, 2), axis=1)
    return arr

tipos = {'Sim2':'Onda plana', 'Sim3': 'Onda plana', 'Sim4-v2': "TinyLev", 'Sim5': 'Onda plana -10s','Sim6-v2': "Onda plana - recorte", 'Sim7': "Onda plana - mapa força - recorte", 'Sim8': "Onda plana", 'Sim9': "TinyLev"} #Sim8 e 9 usando mapa de forças

f=40e3 #Hz
dicMeio = Simulador.ar(1)

a = np.array([1,1]) # [mm]
rhoPol = (900*(10**-6)) #[g/mm^3] #Material: PP
m = (a**3*(4*np.pi/3))*rhoPol # [g], densidade do ar vezes seu volume
Lamb=dicMeio["c"]/f


g=-9.81e3

h=0



numeroSim='Sim8'
diretorio ='Sim8'


with open(f'{diretorio}\\força', 'rb') as dbfile:
    dado = pickle.load(dbfile)
    dbfile.close()
    
Fz0 = dado[0]
Fz1 = dado[1]

def forcas(x):
    return Fz0.ev(*x) + (g*m[0]),Fz1.ev(*x) +(g*m[1])


with open(f'{diretorio}\\resultado-final2', 'rb') as dbfile:
    dados = pickle.load(dbfile)
    dbfile.close()

r0s = np.array(dados['r0'])

if  len(np.shape(r0s))==2:
    zs = r0s
else:
    zs = r0s[:,:,2]


arg = np.argsort(zs[:,0])
zs = zs[arg, :]
N = int(np.sqrt(len(zs)))
zs = np.reshape(zs, [N,N, 2])
arg2 = np.argsort(zs[:,:,1], axis=1)
zs = np.take_along_axis(zs, np.expand_dims(arg2, 2), axis=1)
      
Pbats = np.array(dados['Pbat'])
sPbats = np.array(dados['sPbat'])
Poscs = np.array(dados['Posc'])
sPoscs = np.array(dados['sPosc'])
Amaxs = np.array(dados['Amax'])
sAmaxs = np.array(dados['sAmax'])
Amins = np.array(dados['Amin'])
sAmins = np.array(dados['sAmin'])
       
Amaxs = organiza(Amaxs)
Amins = organiza(Amins)
ampB = Amaxs-Amins

f = organiza(1/Pbats)



zero = root(forcas, [0, Lamb/2])

x0 = zero.x
f0z0 =Fz0.partial_derivative(1,0)(*x0)[0,0]
f0z1 =Fz0.partial_derivative(0,1)(*x0)[0,0]
f1z0 =Fz1.partial_derivative(1,0)(*x0)[0,0]
f1z1 =Fz1.partial_derivative(0,1)(*x0)[0,0]

M = np.array( [[f0z0/m[0],f0z1/m[0]],[f1z0/m[1],f1z1/m[1]]])
sistema=linalg.eig(M)
autovetores=sistema[1]
autovalores=sistema[0]

wBat = np.abs(np.sqrt(-autovalores[0])-np.sqrt(-autovalores[1]))

ganho = 0.5
Gamp, Camp = plot(ampB[:,:,0], ampB[:,:,1], tipos[diretorio], label = 'Amplitude de batimento [mm]', tipo = 1)
for i in range(2):
    Gamp[i].plot(*np.transpose([x0 + ganho*autovetores[:, 0], x0 - ganho*autovetores[:, 0]]), linewidth=1, color = 'xkcd:burnt orange', linestyle=':')
    Gamp[i].plot(*np.transpose([x0 + ganho*autovetores[:, 1], x0 - ganho*autovetores[:, 1]]), linewidth=1,  color = 'xkcd:burnt orange',linestyle=':')
    Gamp[i].scatter(*x0,2,  color = 'xkcd:burnt orange')



Gf, Cf = plot(f[:,:,0], f[:,:,1], tipos[diretorio], label='Frequencia de batimento [Hz]', tipo = 1)

residuo =np.sum((zs -np.expand_dims(x0, (0,1)))**2, axis=2)
argresiduo = np.argmin(residuo)

f0bat0 = f[:,:,0].flatten()[argresiduo]
f0bat1 = f[:,:,1].flatten()[argresiduo]

Cf.ax.axvline(wBat/(2*np.pi),color = 'xkcd:burnt orange',linewidth=1)
Cf.ax.plot(f0bat0, 0.5, 'w.', markersize =3 )
Cf.ax.plot(f0bat1, 0.5, 'w.', markersize =3)


for i in range(2):
    Gf[i].scatter(*x0,2,  color = 'xkcd:burnt orange')

print(f'posição de equilíbrio = {x0} mm')
print(f'frequencia de batimento = {wBat/(2*np.pi)} Hz')

