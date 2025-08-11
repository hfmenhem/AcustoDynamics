import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy as np
import matplotlib.ticker as ticker


#Criando gráfico de frequencia
with open('mapa-batimento-plana', 'rb') as dbfile:
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

f = organiza(1/Pbats)
filtro = f<1e3
f = np.where(filtro, f, np.full(np.shape(f), np.nan))
f = f[:,:,0]

fig, ax = plt.subplots(dpi=300)
fig.tight_layout()
ax.set_title('Marcações de \n pontos de amostragem')
ax.set_aspect(1)

t1=np.min(np.where(np.isnan(f),np.inf, f))
t2=np.max(np.where(np.isnan(f),-np.inf, f))
norm = mpl.colors.Normalize(vmin=t1, vmax=t2, clip=False)

psm0 = ax.pcolormesh(zs[:,:,0],zs[:,:,1], f,shading='nearest', norm=norm)

ax.xaxis.set_major_locator(ticker.LinearLocator(2))
ax.xaxis.set_minor_locator(ticker.LinearLocator(11))
ax.yaxis.set_major_locator(ticker.LinearLocator(3))
ax.yaxis.set_minor_locator(ticker.LinearLocator(31))

ax.tick_params(axis="x")
ax.set_xlabel('$z_o$ - partícula A [mm]')
        
ax.set_ylabel('$z_o$ - partícula B [mm]')
ax.tick_params(axis="y", which = 'both', right=True)
fig.colorbar(psm0, ax=ax, location='right', label='Frequência de batimento [Hz]')
    


#Marcando os pontos

pastaLyap = 'LyapunovGrid-v2'


with open(f'{pastaLyap}\\pontos-organizados', 'rb') as dbfile:
    r0lyap = pickle.load(dbfile)
    dbfile.close()
    
pastapoincare = 'PoincareGrid-v2'

with open(f'{pastapoincare}\\pontos-organizados', 'rb') as dbfile:
    r0poincare = pickle.load(dbfile)
    dbfile.close()

cmaplyap = mpl.colormaps['summer']
colorslyap = cmaplyap(np.linspace(1, 0, len(r0lyap[:,0,0])))

cmappoincare = mpl.colormaps['cividis']
colorspoincare = cmappoincare(np.linspace(0, 1, len(r0poincare[0,:,0])))




for i, linhas in enumerate(r0lyap):
    ax.plot(linhas[:,0], linhas[:,1], '-', color = colorslyap[i], linewidth=1)
    
for i, linhas in enumerate(np.transpose(r0poincare, (1,0,2))):
    ax.plot(linhas[:,0], linhas[:,1], '|', color=colorspoincare[i], markersize=5)
plt.show()

    
    