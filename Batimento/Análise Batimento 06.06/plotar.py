import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy as np

pasta='Sim1-v2'
Nharm = 2

with open(f'{pasta}\\resultado-final', 'rb') as dbfile:
    dados = pickle.load(dbfile)
    dbfile.close()

r0s = dados['r0']
freqs = dados['freq']
amps = dados['amp']

freqN=[]
ampsN=[]
for f, a in zip(freqs, amps):
    a = np.where(np.isnan(a),np.full(np.shape(a),-np.inf), a)
    arg = np.argsort(a, axis=1)
    
    freqN.append(np.take_along_axis(f, arg, axis=1)[:,-2:])
    ampsN.append(np.take_along_axis(a, arg, axis=1)[:,-2:])

freqN=np.array(freqN)
ampsN=np.array(ampsN)

fbat = np.abs(freqN[:,:,0]-freqN[:,:,1])
ambbat = np.abs(np.min((ampsN[:,:,0],ampsN[:,:,1]), axis=0))

# fig, ax = plt.figure(dpi=300)
# pcm = ax[0].pcolor(r0s[0,:], r0s[1,:], fbat)