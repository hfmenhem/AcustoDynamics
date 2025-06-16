import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy as np

pasta='Sim2'
#pasta='Sim4'
Nharm = 2

with open(f'{pasta}\\resultado-final', 'rb') as dbfile:
    dados = pickle.load(dbfile)
    dbfile.close()

r0s = np.array(dados['r0'])
freqs = dados['freq']
amps = dados['amp']

freqN=[]
ampsN=[]
for f, a, i in zip(freqs, amps, range(len(freqs))):
    
    teste=(np.expand_dims(f, 1)/np.expand_dims(f, 2))
    print(f)
    print(teste)
    print(r0s[i][:, 2])
    a = np.where(np.isnan(a),np.full(np.shape(a),0), a)
    arg = np.argsort(np.abs(a), axis=1)
    teste=np.take_along_axis(a, arg, axis=1)
    teste=teste[:,-2:]
    
    f = np.take_along_axis(f, arg, axis=1)
    a = np.take_along_axis(a, arg, axis=1)
    
    if np.shape(f)[1]<2:
        nan = (2-np.shape(f)[1])*[[np.nan], [np.nan]]
        f = np.concatenate((f,nan), axis=1)
        a = np.concatenate((a,nan), axis=1)
        print(i)
        print(r0s[i][:, 2])
    if np.any(f[0,-2:]>60):
        print(i)
        print(r0s[i][:, 2])
    
    freqN.append(f[:,-2:])
    ampsN.append(a[:,-2:])

freqN=np.array(freqN)
ampsN=np.array(ampsN)

print(np.where(np.isnan(ampsN)))

#Organizando os dados
zs = r0s[:,:,2]
arg = np.argsort(zs[:,0])
zs = zs[arg, :]
freqN = freqN[arg,:,:]
ampsN = ampsN[arg,:,:]

N = int(np.sqrt(len(zs)))
zs = np.reshape(zs, [N,N, 2])
freqN = np.reshape(freqN, [N,N, 2,2])
ampsN = np.reshape(ampsN, [N,N, 2,2])

arg2 = np.argsort(zs[:,:,1], axis=1)
zs = np.take_along_axis(zs, np.expand_dims(arg2, 2), axis=1)
freqN = np.take_along_axis(freqN, np.expand_dims(arg2, (2,3)), axis=1)
ampsN = np.take_along_axis(ampsN, np.expand_dims(arg2, (2,3)), axis=1)


fbat = np.abs(freqN[:,:,:,0]-freqN[:,:,:,1])
ambbat = np.abs(np.min((np.abs(ampsN[:,:,:,0]),np.abs(ampsN[:,:,:,1])), axis=0))
minbat = np.abs(np.abs(ampsN[:,:,:,0]) - np.abs(ampsN[:,:,:,1]))
maxbat = np.abs(ampsN[:,:,:,0]) + np.abs(ampsN[:,:,:,1])

filtroA = ambbat<1
filtroF = fbat<10

ambbatF = np.where(filtroA, ambbat, np.full(np.shape(ambbat), np.nan))
fbatF = np.where(filtroF, fbat, np.full(np.shape(fbat), np.nan))

filtroerro = freqN>60
erroF = np.where(filtroerro, freqN, np.full(np.shape(freqN), np.nan))

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

#plot(ambbatF[:,:,0], ambbatF[:,:,1], 'Amplitude de batimento')
plot(fbatF[:,:,0], fbatF[:,:,1], 'Frequencia de batimento')

plot(np.abs(ampsN[:,:,0,0]), np.abs(ampsN[:,:,1,0]), 'amplitudes harmônicos', subt=['harmonico 1', 'harmonico 2'])
plot(minbat[:,:,0],maxbat[:,:,0], 'máximo e mínimo do envelope', subt=['máximo', 'mínimo'])
plot(np.abs(freqN[:,:,0,0]), np.abs(freqN[:,:,1,0]), 'frêquencia harmônicos', subt=['harmonico 1', 'harmonico 2'])
plot(np.abs(erroF[:,:,0,0]), np.abs(erroF[:,:,1,0]), 'frêquencia harmônicos errado', subt=['harmonico 1', 'harmonico 2'])


# n0 = np.where(np.any(freqN[:,:,0,:]==np.nan, axis=2),1, 0)
# n1 = np.where(np.any(freqN[:,:,1,:]==np.nan, axis=2),1, 0)
# plot(n0,n1, 'Frequencia nan')

# n0 = np.where(np.any(ampsN[:,:,0,:]==np.nan, axis=2),1, 0)
# n1 = np.where(np.any(ampsN[:,:,1,:]==np.nan, axis=2),1, 0)
# plot(n0,n1, 'Amplitude nan')



plt.show()