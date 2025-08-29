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
    tpic = np.squeeze(dado['tpic'])
    rpic = dado['rpic']
    vpic = dado['vpic']
    
    
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
        plt.plot(tpic,rpic[i, :], '.')
        
        branch1 = vpic[1, :]>0
        branch2 = vpic[1, :]<0
        
        plt.plot(tpic[branch1],rpic[i, branch1], '.')
        plt.plot(tpic[branch2],rpic[i, branch2], '.')
        
        plt.xlim(0,1)
        #plt.ylim(-10,10)
        
        intf = make_interp_spline(tpic,rpic[i, :])
    
    b1 = rpic[:, vpic[1, :]>0]
    
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('z0(N) [mm]')
    ax.set_ylabel('z1(N) [mm]')
    ax.set_zlabel('z0(N+1) [mm]')

    ax.scatter(b1[0,:-1], b1[1,:-1], b1[0,1:])
    ax.plot_surface(np.array([[np.min(b1[0,:-1]), np.max(b1[0,:-1])],[np.min(b1[0,:-1]), np.max(b1[0,:-1])]]),np.array([[np.min(b1[1,:-1]), np.min(b1[1,:-1])],[np.max(b1[1,:-1]), np.max(b1[1,:-1])]]), np.array([[np.min(b1[0,:-1]), np.max(b1[0,:-1])],[np.min(b1[0,:-1]), np.max(b1[0,:-1])]]))
    # ax.scatter(rpic[0,:-1], rpic[1,:-1], rpic[0,1:])
    # ax.plot_surface(np.array([[np.min(rpic[0,:-1]), np.max(rpic[0,:-1])],[np.min(rpic[0,:-1]), np.max(rpic[0,:-1])]]),np.array([[np.min(rpic[1,:-1]), np.min(rpic[1,:-1])],[np.max(rpic[1,:-1]), np.max(rpic[1,:-1])]]), np.array([[np.min(rpic[0,:-1]), np.max(rpic[0,:-1])],[np.min(rpic[0,:-1]), np.max(rpic[0,:-1])]]))
    
    ax.view_init(5, -90, 0)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('z0(N) [mm]')
    ax.set_ylabel('z1(N) [mm]')
    ax.set_zlabel('z1(N+1) [mm]')

    ax.scatter(b1[0,:-1], b1[1,:-1], b1[1,1:])
    ax.plot_surface(np.array([[np.min(b1[0,:-1]), np.max(b1[0,:-1])],[np.min(b1[0,:-1]), np.max(b1[0,:-1])]]),np.array([[np.min(b1[1,:-1]), np.min(b1[1,:-1])],[np.max(b1[1,:-1]), np.max(b1[1,:-1])]]), np.array([[np.min(b1[1,:-1]), np.min(b1[1,:-1])],[np.max(b1[1,:-1]), np.max(b1[1,:-1])]]))
    # ax.scatter(rpic[0,:-1], rpic[1,:-1], rpic[1,1:])
    # ax.plot_surface(np.array([[np.min(rpic[0,:-1]), np.max(rpic[0,:-1])],[np.min(rpic[0,:-1]), np.max(rpic[0,:-1])]]),np.array([[np.min(rpic[1,:-1]), np.min(rpic[1,:-1])],[np.max(rpic[1,:-1]), np.max(rpic[1,:-1])]]), np.array([[np.min(rpic[1,:-1]), np.min(rpic[1,:-1])],[np.max(rpic[1,:-1]), np.max(rpic[1,:-1])]]))
    
    ax.view_init(5, 0, 0)
        
      
    #resultado = {'r0': r0, 'Pbat': Pbat, 'sPbat': sPbat, 'Posc': Posc, 'sPosc': sPosc, "Amax": Amax, "sAmax": sAmax, "Amin": Amin, "sAmin": sAmin}
    resultado =0
    return resultado





if __name__ == '__main__':
    
    pasta='Poincare-yAeq'
    #dados = 114 # 18
    #dados = 140 # 47
    #dados = 0 # 0
    #dados = 210 #Exemplo de pequenas oscilações
    
    dados = 0 # Caso interessante
    dd = 0
    dados +=dd
    
    #dados = 3506 # 
    
    caminho = f'{pasta}\\dado-{dados}'
    resultado0 = analise(caminho)
    
    
   

