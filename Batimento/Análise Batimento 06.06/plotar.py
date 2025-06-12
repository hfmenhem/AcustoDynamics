import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy as np

pasta='Sim1-v2'
pasta='Sim2'
Nharm = 2

with open(f'{pasta}\\resultado-final', 'rb') as dbfile:
    dados = pickle.load(dbfile)
    dbfile.close()

r0s = np.array(dados['r0'])
freqs = dados['freq']
amps = dados['amp']

freqN=[]
ampsN=[]
for f, a in zip(freqs, amps):
    a = np.where(np.isnan(a),np.full(np.shape(a),0), a)
    arg = np.argsort(np.abs(a), axis=1)
    teste=np.take_along_axis(a, arg, axis=1)
    teste=teste[:,-2:]
    
    freqN.append(np.take_along_axis(f, arg, axis=1)[:,-2:])
    ampsN.append(np.take_along_axis(a, arg, axis=1)[:,-2:])

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



def plot(var0, var1, titulo):

    fig, ax = plt.subplots(1,2,dpi=300)
    fig.suptitle(titulo)
    ax[0].set_title('partícula 0')
    ax[1].set_title('partícula 1')
    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
    psm0 = ax[0].pcolor(zs[:,:,0],zs[:,:,1], var0,shading='nearest')
    psm1 = ax[1].pcolor(zs[:,:,0],zs[:,:,1], var1,shading='nearest')
    fig.colorbar(psm0, ax=ax[0])
    fig.colorbar(psm1, ax=ax[1])

    plt.show()

plot(ambbat[:,:,0], ambbat[:,:,1], 'Amplitude de batimento')
plot(fbat[:,:,0], fbat[:,:,1], 'Frequencia de batimento')

n0 = np.where(np.any(freqN[:,:,0,:]==np.nan, axis=2),1, 0)
n1 = np.where(np.any(freqN[:,:,1,:]==np.nan, axis=2),1, 0)
plot(n0,n1, 'Frequencia nan')

n0 = np.where(np.any(ampsN[:,:,0,:]==np.nan, axis=2),1, 0)
n1 = np.where(np.any(ampsN[:,:,1,:]==np.nan, axis=2),1, 0)
plot(n0,n1, 'Amplitude nan')



plt.show()