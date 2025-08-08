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
    
    pasta='Poincare-yAeq-L'
    
    nomes =[]
    nomesL =[]
    for x in os.listdir(pasta):
        if 'dado-' in x:
            nomes.append(f'{pasta}\\{x}')
        if 'dadoL-' in x:
            nomesL.append(f'{pasta}\\{x}')  
            
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    
    lyap = []
    za = []
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
        dif = np.abs(reventL[1,:n]-revent[1,:n])
        
        dd = np.max(np.abs(r0-r0L))
        serie = np.log(dif)
        Ns = np.arange(n)
        def fit(x, a,b,xlin):
            return np.where( np.less(x,xlin), (a*x)+b, (a*xlin)+b)
        
        
        popt, pcov = curve_fit(fit, Ns, serie, p0=[1, 1e-8, 200], bounds=[[-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]])
        
        print(f'delta fit {np.e**popt[1]} mm')
        print(f'delta {dd} mm')
        print(f't saturamento {popt[2]} ')
        
        # plt.figure(dpi=300)
        # plt.title(f'fit expoente de Lyapunov \n ($z_a, z_b$) = ({r0[0]:.2f}, {r0[1]:.2f}) '+r'$\delta_0$ = ' + f'{dd:.0e} mm')
        # plt.semilogy(Ns,dif, '.', label = 'dados',markersize=1)

        # plt.semilogy(Ns, np.e**(fit(Ns, *popt)) , label=r'$\delta_0 e^{\lambda N}$, $\lambda$ = ' + f'{popt[0]:.2e},\n' + r'$\delta_0$ = ' + f'{np.e**popt[1]:.2e} mm,  N<{popt[2]:.1f} ')
        # plt.legend()
        # plt.ylabel(r'$\delta (N)$ [mm]')
        # plt.xlabel(r'N')
        
        
        
        limit = int(len(revent[0,:])*3/4)
        limit = -100
        #limit = 0
        ax.plot(revent[1,limit:], vevent[1,limit:], label=f'$z_A$ = {r0[0]:.2f} mm', linestyle = '', marker='.', markersize =1)
        
        lyap.append(popt[0])
        za.append(r0[0])
    
    ax.set_xlabel('$z_B$ [mm]')
    ax.set_ylabel('$v_B$ [mm/s]')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    ax.plot(za, lyap, '.')
    ax.set_xlabel('$z_A$ [mm]')
    ax.set_ylabel('$\lambda$ ')

