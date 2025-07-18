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
        plt.xlabel('tempo [s]')
        plt.ylabel('z [mm]')
        plt.plot(tl, intbasico(tl), '-')
        plt.plot(t, rs[i, :], '.')
        plt.plot(tl[indpicM], rsinterpolado[indpicM], '.')
        
        intf = make_interp_spline(tl[indpicM], rsinterpolado[indpicM])
        tint = t[np.logical_and(t<tl[indpicM][-2], t>tl[indpicM][1])]#Os valores analisados para saber se há pico são aqueles que estão entre pontos reais que não são os da borda. Ou seja, não serão considerados válidos picos entre o primeiro e segundo valores usados na interpolação, nem entre os últimos dois, pois nesses intervalos po=dem haver efeitos da interpolação que gerariam falsos picos
        plt.plot(tint, intf(tint), '-')
        
        indpicMM = sig.find_peaks(intf(tint))[0]
        indpicMm = sig.find_peaks(-1*intf(tint))[0]
        plt.plot(tint[indpicMM], intf(tint[indpicMM]), '.')
        plt.plot(tint[indpicMm], intf(tint[indpicMm]), '.')
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
    
    pasta='Sim11'
    nome = 'Sim11'
    
     
    #dados = 114 # 18
    #dados = 140 # 47
    #dados = 0 # 0
    #dados = 210 #Exemplo de pequenas oscilações
    
    dados = 0 # Caso interessante
    dd = 87 
    dados +=dd
    
    #dados = 3506 # 
    
    caminho = f'{pasta}\\{nome}-dado-{dados}'
    nomeSalvar =f'{pasta}-dado-{dados}.csv'
    
    analise(caminho)
    
    with open(caminho, 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dbfile.close()
        
    t = np.expand_dims(dado['t'], 0)
    rs = dado['rs']
    r0 = dado['r0']
    
    salvar = np.transpose(np.concatenate([t, rs], axis=0))
    np.savetxt(nomeSalvar, salvar, delimiter=',', header='tempo [s], z0 [mm], z1 [mm]')
    print(f'{nomeSalvar} salvo!')
    
   

