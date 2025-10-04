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
    
       
    # #===Região próxima da possível separatriz, V2
    pasta='esp-lyapunov'
    n0 = 5
    n1 = 5
    lista = [ (4,0),(0,4),(4,2), #circulos
               (1,4), #direita
             (4,1),(3,0),(2,4),(3,3),(4,3),(1,2),(3,1),(2,0),#cima-baixo
             (3,2),#esquerda
             
             #(0,0),(0,3),
             
             ]
  

    listacor = [9,10, 8,
                18,
                15,14,13,12,4,5,6,7,
                0,
         
                #0,1,
                ]
    
    #
    #===Região próxima da possível separatriz, - as que convergem independentemente do integrador V2
    # pasta='esp-lyapunov'
    # n0 = 5
    # n1 = 5
    # lista = [ (4,0),(0,4),(4,2)] #circulos 
    # listacor = [9,10, 8]
    
    cmappoincare = mpl.colormaps['viridis']
    
    cmapEspecial = mpl.colormaps['tab20c']
    especial = [ (4,0),(0,4),(4,2)]
    
    mesmaCor = 1/((2*n0))
    cores = []
    for d in range(n1):
        shiftCor = mesmaCor*(d/(n1-1))
        teste=np.linspace(shiftCor + mesmaCor, 1-mesmaCor+shiftCor, n0)
        colors = cmappoincare(teste)
        cores.append(colors)
        
    for i, corEsp in enumerate(especial):
        cores[corEsp[0]][corEsp[1]] = cmapEspecial((16+i)/20)        
    
     
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
    
    nomesOrg = np.array(nomes)[inds.flatten()]


    
    figs = []
    axs=[]
    for i in range(4):
        figi = plt.figure(dpi=300, figsize=(10,7.5))
        figs.append(figi)
        axi = figi.add_subplot()
        axi.set_xlabel(r'$t$ [s]')
        axi.set_ylabel(f'$X_{i+1}$ [Hz]')
        axi.set_title(f'Expoente de Lyapunov {i+1}')
        axs.append(axi)
        axi.grid()
   
    # fig5 = plt.figure(dpi=300)
    # ax5 = fig5.add_subplot()
    # ax5.set_xlabel(r'$t$ [s]')
    # ax5.set_ylabel(r'$soma X_i$ [Hz]')
    # ax5.set_title('Soma dos expoentes de Lyapunov')
    
   # for i, coord in enumerate(lista):
        
    for d in range(n1):
        display = nomesOrg[d*n0:(d+1)*n0]
        
        for i, nome in enumerate(display):
            
      
    
            with open(nome, 'rb') as dbfile:
                dado = pickle.load(dbfile)
                dbfile.close()
            r0 = dado['r0']
            xi = dado['Xi']
            ts = dado['ts']
            
            descarte = int(0.2*len(ts))
            for j, res in enumerate(np.transpose(xi)):
                axs[j].plot(ts[descarte:], res[descarte:], '-', color = cores[d][i])
                
           #ax5.plot(ts[descarte:], np.sum(xi, axis=1)[descarte:], '-', color = cores[d][i])



    plt.show()
        


        


    
   
