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

def analise(nome):
    with open(nome, 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dbfile.close()
        
    t = dado['t']
    rs = dado['rs']
    r0 = dado['r0']
    tevent = np.squeeze(dado['tpic'])
    revent = dado['rpic']
    vevent = dado['vpic']
    
    
    apenasz = len(np.shape(rs))==2 #booleano indicando se temos apenas as componentes em Z (True) ou se temos todas as 3 componentes(False) 
    if not apenasz:
        r0 = r0[:,2]
        rs = rs[:, :, 2]

    print(f'[{r0[0]:.13e}, {r0[1]:.13e}]')

    Npar = 2
    for i in range(Npar):
        plt.figure(dpi=300)
        plt.title(f'sinal-{i}')
        plt.xlabel('tempo [s]')
        plt.ylabel('z [mm]')

        plt.plot(t, rs[i, :], '-')
        plt.plot(tevent,revent[i, :], '.')
        

        plt.plot(tevent,revent[i, :], '.')
        plt.plot(tevent,revent[i, :], '.')
        
        plt.xlim(0,1)
        #plt.ylim(-10,10)
        
        
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    ax.set_xlabel('$z_B$ [mm]')
    ax.set_ylabel('$v_B$ [mm/s]')
    
    ax.plot(revent[1,:], vevent[1,:], linestyle = '', marker='.')
    plt.show()

    

  
      
    #resultado = {'r0': r0, 'Pbat': Pbat, 'sPbat': sPbat, 'Posc': Posc, 'sPosc': sPosc, "Amax": Amax, "sAmax": sAmax, "Amin": Amin, "sAmin": sAmin}
    resultado =0
    return resultado





if __name__ == '__main__':
    
    pasta='Poincare-yAeq-teste'
    
    nomes =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            nomes.append(f'{pasta}\\{x}')
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    
    for nome in nomes:
        with open(nome, 'rb') as dbfile:
            dado = pickle.load(dbfile)
            dbfile.close()
            
     
        r0 = dado['r0']
        revent = dado['rpic']
        vevent = dado['vpic']
        
        ax.set_xlabel('$z_B$ [mm]')
        ax.set_ylabel('$v_B$ [mm/s]')
        
        limit = int(len(revent[0,:])*3/4)
        limit = -100
        #limit = 0
        ax.plot(revent[1,limit:], vevent[1,limit:], label=f'$z_A$ = {r0[0]:.2f} mm', linestyle = '', marker='.', markersize =1)
    
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
   

