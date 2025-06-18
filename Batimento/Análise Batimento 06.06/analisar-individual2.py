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
    
    apenasz = len(np.shape(rs))==2 #booleano indicando se temos apenas as componentes em Z (True) ou se temos todas as 3 componentes(False) 
    if not apenasz:
        r0 = r0[:,2]
        rs = rs[:, :, 2]

    print(r0[:])


    Npar = 2
    
    Pbat=[]
    sPbat=[]
    Posc=[]
    sPosc=[]
    Amax = []
    sAmax = []
    Amin = []
    sAmin = []
    for i in range(Npar):
        intbasico = make_interp_spline(t, rs[i, :])
        tl=np.linspace(0, t[-1], 100*len(t))
        rsinterpolado = intbasico(tl)
        
        indpicM = sig.find_peaks(rsinterpolado)[0]
        #indpicm = sig.find_peaks(-1*rsinterpolado)[0]
        
        plt.figure(dpi=300)
        plt.title(f'sinal-{i}')
        plt.plot(tl, intbasico(tl), '-')
        plt.plot(t, rs[i, :], '.')
        plt.plot(tl[indpicM], rsinterpolado[indpicM], '.')
        
        intf = make_interp_spline(tl[indpicM], rsinterpolado[indpicM])
        plt.plot(t, intf(t), '-')
        
        indpicMM = sig.find_peaks(intf(t))[0]
        indpicMm = sig.find_peaks(-1*intf(t))[0]
        plt.plot(t[indpicMM], intf(t[indpicMM]), '.')
        plt.plot(t[indpicMm], intf(t[indpicMm]), '.')
        # if i==0:
        #     plt.ylim((0.79, 0.794))
        # else:
        #     plt.ylim((4.39, 4.394))
        #períodos
        
        d=np.diff(tl[indpicM])
        Posc.append(np.mean(d))
        sPosc.append(np.std(d)/np.sqrt(len(d)))
        if len(indpicMM)>1:
            d=np.diff(t[indpicMM])
            Pbat.append(np.mean(d))
            sPbat.append(np.std(d)/np.sqrt(len(d)))
        elif len(indpicMm)>1:
            d=np.diff(t[indpicMm])
            Pbat.append(np.mean(d))
            sPbat.append(np.std(d)/np.sqrt(len(d)))
        elif len(indpicMm) == 1 and len(indpicMM) == 1:
            Pbat.append(2*np.abs(t[indpicMm]-t[indpicMM])[0])
            sPbat.append(np.nan)
        else:
            Pbat.append(np.nan)
            sPbat.append(np.nan)
        
        Amax.append(np.mean(intf(t[indpicMM])))
        sAmax.append(np.std(intf(t[indpicMM]))/len(indpicMM))
        
        Amin.append(np.mean(intf(t[indpicMm])))
        sAmin.append(np.std(intf(t[indpicMm]))/len(indpicMm))
        
        print(Amax[i])
        print(Amin[i])
        
        print(1/Pbat[i])
        
        print(f'Pbat = {Pbat[i]} +- {sPbat[i]} s ')
        print(f'Posc = {Posc[i]} +- {sPosc[i]} s ')

    resultado = {'r0': r0, 'Pbat': Pbat, 'sPbat': sPbat, 'Posc': Posc, 'sPosc': sPosc, "Amax": Amax, "sAmax": sAmax, "Amin": Amin, "sAmin": sAmin}
    return resultado





if __name__ == '__main__':
    
    pasta='Sim7'
    nome = 'Sim7'
    #dados = 114 # 18
    #dados = 140 # 47
    #dados = 0 # 0
    #dados = 210 #Exemplo de pequenas oscilações
    
    dados = 19 # entender
    dados = 40 # entender
    #dados = 3506 # 
    
    caminho = f'{pasta}\\{nome}-dado-{dados}'
    
    resultado = analise(caminho)
    
    
    
    # for a, f, i in zip(resultado['amp'],resultado['freq'], range(2)):
    #     a = np.where(np.isnan(a),np.full(np.shape(a),0), a)
    #     arg = np.argsort(np.abs(a), axis=0)
       
        
    #     f = np.take_along_axis(np.array(f), arg, axis=0)
    #     a = np.take_along_axis(np.array(a), arg, axis=0)
        
        
    #     print(i)
    #     print('f')
    #     print(f)
    #     print('a')
    #     print(np.abs(a))


