import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

from main import Simulador
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft, fftfreq
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline

import pickle
import concurrent.futures

if __name__ == '__main__':
    
    pasta='LyapunovGrid-v2'
    n0 = 60
    n1 = 4
    
    
    nomes =[]
    nomesL =[]
    for x in os.listdir(pasta):
        if 'dado-' in x:
            nomes.append(f'{pasta}\\{x}')
        if 'dadoL-' in x:
            nomesL.append(f'{pasta}\\{x}')  
            

    lyap = []
    r0s= []
    for nome, nomeL in zip(nomes, nomesL):
        with open(nome, 'rb') as dbfile:
            dado = pickle.load(dbfile)
            dbfile.close()
        with open(nomeL, 'rb') as dbfile:
            dadoL = pickle.load(dbfile)
            dbfile.close()
            
     
        r0 = dado['r0']
        revent = dado['rpic']
        vevent = dado['vpic']
        
        r0L = dadoL['r0']
        reventL = dadoL['rpic']
        veventL = dadoL['vpic']
        
        n = np.min([len(reventL[1,:]),len(revent[1,:])])
        print(f'valor de N máximo {n}')
        dif = np.abs(reventL[1,:n]-revent[1,:n])
        
        dd = np.max(np.abs(r0-r0L))
        serie = np.log(dif)
        Ns = np.arange(n)
        def fit(x, a,b,xlin):
            return np.where( np.less(x,xlin), (a*x)+b, (a*xlin)+b)
        
        
        popt, pcov = curve_fit(fit, Ns, serie, p0=[1, 1e-8, 200], bounds=[[-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]])
        
        # print(f'delta fit {np.e**popt[1]} mm')
        # print(f'delta {dd} mm')
        # print(f't saturamento {popt[2]} ')
        
        lyap.append(popt[0])
        r0s.append(r0)
    
    r0s = np.array(r0s)
    lyap = np.array(lyap)
    
    arg = np.argsort(r0s[:,1])
    r0s = r0s[arg, :]
    lyap = lyap[arg]

    r0s = np.reshape(r0s, [n1,n0, 2])
    lyap = np.reshape(lyap, [n1,n0])

    arg2 = np.argsort(r0s[:,:,0], axis=1)
    r0s = np.take_along_axis(r0s, np.expand_dims(arg2, 2), axis=1)
    lyap = np.take_along_axis(lyap, arg2, axis=1)
    
    with open(f'{pasta}\\pontos-organizados', 'wb') as dbfile:
        pickle.dump(r0s, dbfile)
        dbfile.close()
    
    cmap = mpl.colormaps['summer']
    colors = cmap(np.linspace(1, 0, len(r0s[:,0,0])))
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    ax.tick_params(axis="x", which = 'minor', bottom=False)
    ax.minorticks_on()
    plt.title('Expoente de Lyapunov em função de $z_{o,A}$ e $z_{o,B}$')
    for i, zb in enumerate(r0s[:,0,1]):
        ax.plot(r0s[i,:,0], lyap[i,:], '-', label = '$z_{o,B}$ = '+ f'{zb:.2f} mm', color =colors[i])
        ax.set_xlabel('$z_{o,A}$ [mm]')
        ax.set_ylabel('$\lambda$')
        
    plt.legend()
    plt.show()
    
    

        
        
        
        

