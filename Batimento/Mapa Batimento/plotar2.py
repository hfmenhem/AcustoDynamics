import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy as np
import matplotlib.ticker as ticker

pasta='Sim8'
#pasta='Sim4'
Nharm = 2

with open(f'{pasta}\\resultado-final2', 'rb') as dbfile:
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

def plot(var0, var1, titulo, subt = None, label='', tipo = 1):

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
    
    psm0 = ax[0].pcolormesh(zs[:,:,0],zs[:,:,1], var0,shading='nearest', norm=norm)
    psm1 = ax[1].pcolormesh(zs[:,:,0],zs[:,:,1], var1,shading='nearest', norm=norm)

    ax[0].plot(-0.1, 4.6)
    
    for i in range(2):
        if tipo == 1:
            ax[i].xaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].xaxis.set_minor_locator(ticker.LinearLocator(31))
            ax[i].yaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].yaxis.set_minor_locator(ticker.LinearLocator(31))
            
            ax[i].tick_params(axis="x", labelrotation=0)
            ax[i].set_xlabel('$z_o$ - partícula 0 [mm]')
        elif tipo == 2:
            ax[i].xaxis.set_major_locator(ticker.LinearLocator(2))
            ax[i].xaxis.set_minor_locator(ticker.LinearLocator(11))
            ax[i].yaxis.set_major_locator(ticker.LinearLocator(3))
            ax[i].yaxis.set_minor_locator(ticker.LinearLocator(31))
            
            ax[i].tick_params(axis="x", labelrotation=-10)
            ax[i].set_xlabel('$z_o$ - partícula 0 [mm]')
            
    
    if tipo == 1:
        ax[0].set_ylabel('$z_o$ - partícula 1 [mm]')
        ax[1].tick_params(axis="y", which = 'both', labelleft=False, right=True)
        ax[0].tick_params(axis="y", which = 'both', right=True)
        fig.colorbar(psm1, ax=ax, location='bottom', pad=0.21, label=label)
    elif tipo == 2:
        ax[0].set_ylabel('$z_o$ - partícula 1 [mm]')
        ax[1].tick_params(axis="y", which = 'both', labelleft=False, right=True)
        ax[0].tick_params(axis="y", which = 'both', right=True)
        fig.colorbar(psm1, ax=ax, location='right', label=label)
        
    plt.show()
    
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

      
Pbats = np.array(dados['Pbat'])
sPbats = np.array(dados['sPbat'])
Poscs = np.array(dados['Posc'])
sPoscs = np.array(dados['sPosc'])
Amaxs = np.array(dados['Amax'])
sAmaxs = np.array(dados['sAmax'])
Amins = np.array(dados['Amin'])
sAmins = np.array(dados['sAmin'])
       

f = organiza(1/Pbats)
filtro = f<1e3
f = np.where(filtro, f, np.full(np.shape(f), np.nan))

Amaxs = organiza(Amaxs)
Amins = organiza(Amins)
ampB = Amaxs-Amins

filtro = ampB>0.8
filtro = np.logical_and(filtro, np.expand_dims(zs[:,:,0],2)<-0.2)
filtro = np.logical_and(filtro, np.expand_dims(zs[:,:,1],2)<3.5)

ampBf = np.where(filtro, ampB, np.full(np.shape(ampB), np.nan))

print(np.transpose((zs[:,:,0][filtro[:,:,0]], zs[:,:,1][filtro[:,:,0]])))

tipos = {'Sim3': 'Onda plana', 'Sim4-v2': "TinyLev", 'Sim6-v2': "Onda plana - recorte", 'Sim7': "Onda plana - Teste mapa força - recorte", 'Sim8': "Onda plana - Teste mapa força"} 


plot(ampB[:,:,0], ampB[:,:,1], tipos[pasta], label = 'Amplitude de batimento [mm]', tipo = 1)
plot(ampBf[:,:,0], ampBf[:,:,1], tipos[pasta], label = 'Amplitude de batimento [mm]', tipo = 1)

plot(f[:,:,0], f[:,:,1], tipos[pasta], label='Frequencia de batimento [Hz]', tipo = 1)
# plot(ampB[:,:,0], ampB[:,:,1], tipos[pasta], label = 'Amplitude de batimento [mm]', tipo = 2)
# plot(Amaxs[:,:,0], Amaxs[:,:,1], tipos[pasta], label = 'Amplitude máxima [mm]', tipo = 2)
# plot(Amins[:,:,0], Amins[:,:,1], tipos[pasta], label = 'Amplitude mínima [mm]', tipo = 2)

