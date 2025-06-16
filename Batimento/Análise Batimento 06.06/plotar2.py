import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy as np

pasta='Sim6-v2'
#pasta='Sim4'
Nharm = 2

with open(f'{pasta}\\resultado-final2', 'rb') as dbfile:
    dados = pickle.load(dbfile)
    dbfile.close()

r0s = np.array(dados['r0'])
zs = r0s[:,:,2]
arg = np.argsort(zs[:,0])
zs = zs[arg, :]
N = int(np.sqrt(len(zs)))
zs = np.reshape(zs, [N,N, 2])
arg2 = np.argsort(zs[:,:,1], axis=1)
zs = np.take_along_axis(zs, np.expand_dims(arg2, 2), axis=1)

def plot(var0, var1, titulo, subt = None):

    fig, ax = plt.subplots(1,2,dpi=300)
    fig.suptitle(titulo)
    if subt is None:
        ax[0].set_title('partícula 0')
        ax[1].set_title('partícula 1')
    else:
        ax[0].set_title(subt[0])
        ax[1].set_title(subt[1])
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    
    t1=np.min(np.where(np.isnan([var0, var1]),np.inf, [var0, var1]))
    t2=np.max(np.where(np.isnan([var0, var1]),-np.inf, [var0, var1]))
    norm = mpl.colors.Normalize(vmin=t1, vmax=t2, clip=False)
    
    psm0 = ax[0].pcolor(zs[:,:,0],zs[:,:,1], var0,shading='nearest', norm=norm)
    psm1 = ax[1].pcolor(zs[:,:,0],zs[:,:,1], var1,shading='nearest', norm=norm)
    fig.colorbar(psm0, ax=ax[0])
    fig.colorbar(psm1, ax=ax[1])

    plt.show()
    
#Organizando os dados
def organiza(arr):
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

      
Pbats = np.array(dados['Pbat'])
sPbats = np.array(dados['sPbat'])
Poscs = np.array(dados['Posc'])
sPoscs = np.array(dados['sPosc'])
Amaxs = np.array(dados['Amax'])
sAmaxs = np.array(dados['sAmax'])
Amins = np.array(dados['Amin'])
sAmins = np.array(dados['sAmin'])
       

f = organiza(1/Pbats)
Amaxs = organiza(Amaxs)
Amins = organiza(Amins)
ampB = Amaxs-Amins



plot(f[:,:,0], f[:,:,1], 'Frequencia de batimento')
plot(ampB[:,:,0], ampB[:,:,1], 'Amplitude de batimento')
plot(Amaxs[:,:,0], Amaxs[:,:,1], 'Amplitude máxima')
plot(Amins[:,:,0], Amins[:,:,1], 'Amplitude mínima')

