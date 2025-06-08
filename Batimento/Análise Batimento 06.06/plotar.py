import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft, fftfreq
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import find_peaks
from scipy.signal.windows import blackman
from scipy.stats import gmean
from scipy.optimize import curve_fit

def senos(t, med, *kwarg):
    t = np.expand_dims(t, 0)
    N = len(kwarg)//3
    amp = np.expand_dims(np.array(kwarg[0:N]),1)
    freq = np.expand_dims(np.array(kwarg[N:2*N]),1)
    fase = np.expand_dims(np.array(kwarg[2*N:3*N]),1)
    return med+np.sum((amp*np.sin((2*np.pi*freq*t)+fase)), axis=0)

def exp(t, a, b):
    return a*(np.e**b)


rs = np.load('rs2.npy')
vs = np.load('vs2.npy')
t = np.load('t2.npy')

Npar=len(rs[:,0,0])
dt = t[1]-t[0]

fig, axs = plt.subplots(Npar, 1, sharex=True,dpi=300)
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

fig2, axs2 = plt.subplots(Npar, 1, sharex=True,dpi=300)
fig2.subplots_adjust(hspace=0.05)  # adjust space between Axes

yfs=[]
xfs=[]
flim=0
Nsenos = 5
for i in range(Npar):
    N=len(rs[i, :, 2])
    w = blackman(N)
    yf= (2.0/N)*np.abs(fft(rs[i, :, 2]*w)[0:N//2])
    xf=fftfreq(N, dt)[:N//2]
    
    bspl = Akima1DInterpolator(xf, yf)
    
    #bspl = make_interp_spline(xf, yf, k=3)
    xinter = np.linspace(np.min(xf),np.max(xf) , 20*N)
    yinter = bspl(xinter )
    
    indpic = find_peaks(yinter)[0]
    xfpic = xinter[indpic]
    yfpic = yinter[indpic]
    

    
    # indpic = find_peaks(yf)[0]
    # xfpic = xf[indpic]
    # yfpic = yf[indpic]
    
    indy = np.argsort(yfpic)
    xfpic = np.flip(xfpic[indy])
    yfpic = np.flip(yfpic[indy])
    
    
    # indy = np.argsort(xfpic)
    # xfpic =xfpic[indy]
    # yfpic =yfpic[indy]

    
    p0=[np.mean(rs[i, :, 2]), *yfpic[:Nsenos], *(xfpic[:Nsenos]), *np.zeros(Nsenos)]
    
    popt, pcov = curve_fit(senos, t, rs[i, :, 2], p0)
    
        
    
    print(i)
    #print(p0)
    print(popt)
    print(np.linalg.cond(pcov))
    
    axs2[i].semilogy(xf, yf, '.')
    axs2[i].semilogy(xinter, yinter, '-')
    axs2[i].semilogy(xfpic, yfpic, '.')
    axs2[i].semilogy(xfpic[:Nsenos], yfpic[:Nsenos], '.')
    #axs2[i].set_xlim(0,150)
    
    axs[i].plot(t,rs[i, :, 2]-senos(t, *popt), '.')
    
    print(f'razão desvio padrão / menor amplitude utilizada: {np.std(rs[i, :, 2]-senos(t, *popt))/abs(popt[Nsenos])}')
    
    # axs[i].plot(t,senos(t, *popt), '-')
    # axs[i].plot(t,rs[i, :, 2], '.')
    axs[i].grid()
    
   
   
    
fig.supylabel("z [mm]")
fig.supxlabel("t [s]")
fig.suptitle('resíduos da regressão')
