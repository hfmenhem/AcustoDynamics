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
    
    pasta='PoincareGrid-v2'
    n0 = 5
    n1 = 4
    
    nomes =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            nomes.append(f'{pasta}\\{x}')

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
    
    with open(f'{pasta}\\pontos-organizados', 'wb') as dbfile:
        pickle.dump(r0s, dbfile)
        dbfile.close()
    
    cmappoincare = mpl.colormaps['cividis']
    colorspoincare = cmappoincare(np.linspace(0, 1, len(r0s[0,:,0])))
    
    fig, ax = plt.subplots(2,2,dpi=300, figsize=(10,7))
    fig.suptitle('Espaço de fase da seção de Poincaré')
    fig.tight_layout()
    plt.subplots_adjust(wspace=.35, hspace=.18)
    ax = ax.flatten()
    for i, indi in enumerate( inds):
        for j , indij in enumerate(indi) :
            with open(nomes[indij], 'rb') as dbfile:
                dado = pickle.load(dbfile)
                dbfile.close()
                
         
            r0 = dado['r0']
            revent = dado['rpic']
            vevent = dado['vpic']
            

            #limit = 0
            ax[i].plot(revent[1,:], vevent[1,:], label=f'{r0[0]:.2f}', linestyle = '', marker='.', markersize =1, color = colorspoincare[j])
        
        ax[i].set_title('$z_{o, B}$ = '+f'{r0[1]:.2f} mm')
        ax[i].legend(loc='upper left', bbox_to_anchor=(1, 1), title='$z_{o, A}\, [mm]$')
    
    ax[2].set_xlabel('$z_B$ [mm]')
    ax[3].set_xlabel('$z_B$ [mm]')
    ax[0].set_ylabel('$v_B$ [mm/s]')
    ax[2].set_ylabel('$v_B$ [mm/s]')
    plt.show()
    
   

