import os
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
Amaxs = np.array(dados['Amax'])
Amins = np.array(dados['Amin'])

Amaxs = organiza(Amaxs)
Amins = organiza(Amins)
ampB = Amaxs-Amins

ampB = ampB[:,:,0]


fig, ax = plt.subplots(dpi=300)
fig.tight_layout()
ax.set_title('Marcações de \n pontos de amostragem')
ax.set_aspect(1)

t1=np.min(np.where(np.isnan(ampB),np.inf, ampB))
t2=np.max(np.where(np.isnan(ampB),-np.inf, ampB))
norm = mpl.colors.Normalize(vmin=t1, vmax=t2, clip=False)

psm0 = ax.pcolormesh(zs[:,:,0],zs[:,:,1], ampB,shading='nearest', norm=norm)

ax.xaxis.set_major_locator(ticker.LinearLocator(2))
ax.xaxis.set_minor_locator(ticker.LinearLocator(11))
ax.yaxis.set_major_locator(ticker.LinearLocator(3))
ax.yaxis.set_minor_locator(ticker.LinearLocator(31))

ax.tick_params(axis="x")
ax.set_xlabel('$z_o$ - partícula A [mm]')
        
ax.set_ylabel('$z_o$ - partícula B [mm]')
ax.tick_params(axis="y", which = 'both', right=True)
fig.colorbar(psm0, ax=ax, location='right', label='Amplitude de batimento [mm]')
    


#Marcando os pontos

pastaLyap = 'LyapunovGrid4'


pasta='PoincareGrid4'
nomes =[]
for x in os.listdir(pasta):
    if 'dado' in x:
        nomes.append(f'{pasta}\\{x}')



n0 = 20
n1 = 20

r0s= []

for nome in nomes:
    with open(nome, 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dbfile.close()
    r0 = dado['r0']
    r0s.append(r0)
    
r0s = np.array(r0s)
inds = np.arange(len(r0s))

arg = np.argsort(r0s[:,1])
r0s = r0s[arg, :]
inds = inds[arg]

r0s = np.reshape(r0s, [n1,n0, 2])
inds = np.reshape(inds, [n1,n0])

arg2 = np.argsort(r0s[:,:,0], axis=1)
r0s = np.take_along_axis(r0s, np.expand_dims(arg2, 2), axis=1)
inds = np.take_along_axis(inds, arg2, axis=1)


lista = [(2,0), (2,1),(3,2),#caos?
         (2,2), (7,5), (7,6),#quase fechando loop
         (7,4),(17,0), #dentro
          (17,1), (17,2),#alto batimento
          (16,3),(7,7)#+ próximo do equilibrio?
         
         ]

listacor = [18,17,16,
            13,14,15,
            2,3,
            8,9,
            4,5
            ]

lista = np.array(lista)
pontos = r0s[lista[:,0],lista[:,1]]

print(pontos)
#ax.plot(pontos[:,0], pontos[:,1], '.', color='k', markersize=5, marker='+')

cmap = mpl.colormaps['tab20b']
colors= cmap(np.linspace(0,1, 20))

# cmappoincare = mpl.colormaps['cividis']
# colorspoincare = cmappoincare(np.linspace(0, 1, len(r0poincare[0,:,0])))




for i, pt in enumerate(pontos):
    ax.plot(pt[0], pt[1], '+', color = colors[listacor[i]], markersize=5)
    
# for i, linhas in enumerate(np.transpose(r0poincare, (1,0,2))):
#     ax.plot(linhas[:,0], linhas[:,1], '|', color=colorspoincare[i], markersize=5)
# plt.show()

    
    