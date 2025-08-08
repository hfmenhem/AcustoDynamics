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
    dicionario = {0: 'A', 1: 'B'}
    
    fig, ax = plt.subplots(2,1,dpi=300, figsize=(10,6))
    fig.suptitle('Trajetória das partículas -  ($z_{o, A}, z_{o, B}$) ='+ f' ({r0[0]:.2f}, {r0[1]:.2f}) mm')
    ax[0].set_ylabel('$z_B$ [mm]')
    ax[1].set_ylabel('$z_A$ [mm]')
    ax[1].set_xlabel('tempo [s]')
    ax[0].tick_params(axis="x", which = 'both', labelbottom=False)
    ax[0].tick_params(axis="x", which = 'minor', bottom=True)
    ax[0].minorticks_on()
    ax[1].minorticks_on()
    ax[0].tick_params(axis="y", which = 'minor', left=False)
    ax[1].tick_params(axis="y", which = 'minor', left=False)
    #ax[0].tick_params(axis="y", which = 'both', right=True)
    #ax[0].set_xticks([])
    
    for i in range(Npar):
        intbasico = make_interp_spline(t, rs[i, :])
        tl=np.linspace(0, t[-1], 100*len(t))
        rsinterpolado = intbasico(tl)
        
        indpicM = sig.find_peaks(rsinterpolado)[0]
        #indpicm = sig.find_peaks(-1*rsinterpolado)[0]
        
        # plt.figure(dpi=300)
        # plt.title(f'Trajetória da partícula {dicionario[i]}')
        # plt.xlabel('tempo [s]')
        # plt.ylabel('z [mm]')
        #plt.plot(tl, intbasico(tl), '-')
        ax[-i+1].plot(t, rs[i, :], '.', markersize=1, color='xkcd:beige')
        #plt.plot(tl[indpicM], rsinterpolado[indpicM], '.')
        
        intf = make_interp_spline(tl[indpicM], rsinterpolado[indpicM])
        tint = t[np.logical_and(t<tl[indpicM][-2], t>tl[indpicM][1])]#Os valores analisados para saber se há pico são aqueles que estão entre pontos reais que não são os da borda. Ou seja, não serão considerados válidos picos entre o primeiro e segundo valores usados na interpolação, nem entre os últimos dois, pois nesses intervalos po=dem haver efeitos da interpolação que gerariam falsos picos
        ax[-i+1].plot(tint, intf(tint), '-', color = 'xkcd:navy blue')
        
        indpicMM = sig.find_peaks(intf(tint))[0]
        indpicMm = sig.find_peaks(-1*intf(tint))[0]
        ax[-i+1].plot(tint[indpicMM], intf(tint[indpicMM]), '.', color = 'xkcd:dark teal')
        ax[-i+1].plot(tint[indpicMm], intf(tint[indpicMm]), '.', color = 'xkcd:brick red')
        # if i==0:
        #     plt.ylim((0.79, 0.794))
        # else:
        #     plt.ylim((4.39, 4.394))
        #períodos
        
        d=np.diff(tl[indpicM])
        Posc.append(np.mean(d))
        sPosc.append(np.std(d)/np.sqrt(len(d)))
        
        if max(len(indpicMM), len(indpicMm))>1:
            if len(indpicMM)>=len(indpicMm):
                d=np.diff(tint[indpicMM])
            else:
                d=np.diff(tint[indpicMm])
            Pbat.append(np.mean(d))
            sPbat.append(np.std(d)/np.sqrt(len(d)))
        elif len(indpicMm) == 1 and len(indpicMM) == 1:
            Pbat.append(2*np.abs(tint[indpicMm]-tint[indpicMM])[0])
            sPbat.append(np.nan)
        else:
            Pbat.append(np.nan)
            sPbat.append(np.nan)
        
        Amax.append(np.mean(intf(tint[indpicMM])))
        sAmax.append(np.std(intf(tint[indpicMM]))/len(indpicMM))
        
        Amin.append(np.mean(intf(tint[indpicMm])))
        sAmin.append(np.std(intf(tint[indpicMm]))/len(indpicMm))
        
        print(Amax[i])
        print(Amin[i])
        
        print(1/Pbat[i])
        
        print(f'Pbat = {Pbat[i]} +- {sPbat[i]} s ')
        print(f'Posc = {Posc[i]} +- {sPosc[i]} s ')

    resultado = {'r0': r0, 'Pbat': Pbat, 'sPbat': sPbat, 'Posc': Posc, 'sPosc': sPosc, "Amax": Amax, "sAmax": sAmax, "Amin": Amin, "sAmin": sAmin}
    return resultado





if __name__ == '__main__':
    
    pasta='Sim12'
    nome = 'Sim12'
    #dados = 114 # 18
    #dados = 140 # 47
    #dados = 0 # 0
    #dados = 210 #Exemplo de pequenas oscilações
    
    dados = 9800 # Caso interessante
    dd=78
    dd+=1400
    dados +=dd
    
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


