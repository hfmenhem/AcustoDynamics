import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft, fftfreq
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

rs = np.load('rs.npy')
vs = np.load('vs.npy')
t = np.load('t.npy')

Npar=len(rs[:,0,0])
dt = t[1]-t[0]

plt.figure(dpi=300)
fig, axs = plt.subplots(Npar, 1, sharex=True)
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

yfs=[]
xfs=[]
flim=1e-3
for i in range(Npar):
    N=len(rs[i, :, 2])
    yf=fft(rs[i, :, 2])
    xf=fftfreq(N, dt)[:N//2]
    yfs.append(yf)
    xfs.append(xf)
    
    yplot= 2.0/N * np.abs(yf[0:N//2]) 
    filtro =yplot>flim
    nans=np.full(np.shape(yplot), np.nan)
    
    axs[i].plot(np.where(filtro,xf, nans)[1:], np.where(filtro,yplot, nans)[1:], '.')
    xlim=(30,50)
    axs[i].set_xlim(xlim)
    #axs[i].set_ylim((0,0.4))
    
    bspl = make_interp_spline(xf, yplot, k=3)
    xinter = np.linspace(*xlim , 2000)
    yinter = bspl(xinter )
    #axs[i].plot(xinter, yinter, '--')
    indpic = find_peaks(yplot)[0]
    axs[i].plot(xf[indpic], yplot[indpic], '.')
    
    #axs[i].grid()
    
fig.supylabel("amplitude [mm]")
fig.supxlabel("frequência [Hz]")
fig.suptitle('Transformada de Fourier')
