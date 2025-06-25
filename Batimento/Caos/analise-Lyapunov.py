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
from scipy.stats import linregress

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

    print(f'[{r0[0]:.13e}, {r0[1]:.13e}]')


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
        
        # plt.figure(dpi=300)
        # plt.title(f'sinal-{i}')
        # plt.xlabel('tempo [s]')
        # plt.ylabel('z [mm]')
        # plt.plot(tl, intbasico(tl), '-')
        # plt.plot(t, rs[i, :], '.')
        # plt.plot(tl[indpicM], rsinterpolado[indpicM], '.')
        
        #plt.xlim(0, 1.0)
        
        intf = make_interp_spline(tl[indpicM], rsinterpolado[indpicM])
        tint = t[np.logical_and(t<tl[indpicM][-2], t>tl[indpicM][1])]#Os valores analisados para saber se há pico são aqueles que estão entre pontos reais que não são os da borda. Ou seja, não serão considerados válidos picos entre o primeiro e segundo valores usados na interpolação, nem entre os últimos dois, pois nesses intervalos po=dem haver efeitos da interpolação que gerariam falsos picos
        # plt.plot(tint, intf(tint), '-')
        
        indpicMM = sig.find_peaks(intf(tint))[0]
        indpicMm = sig.find_peaks(-1*intf(tint))[0]
        # plt.plot(tint[indpicMM], intf(tint[indpicMM]), '.')
        # plt.plot(tint[indpicMm], intf(tint[indpicMm]), '.')
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

    resultado = {'r0': r0, 'Pbat': Pbat, 'sPbat': sPbat, 'Posc': Posc, 'sPosc': sPosc, "Amax": Amax, "sAmax": sAmax, "Amin": Amin, "sAmin": sAmin, "envelope": intf, "t": t}
    return resultado





if __name__ == '__main__':
    
    #pasta='SimT-v2'
    pasta='Sim-Lyap-8'
    #pasta='SimF-v6'
    #dados = 114 # 18
    #dados = 140 # 47
    #dados = 0 # 0
    #dados = 210 #Exemplo de pequenas oscilações
    
    dados = 0 # Caso interessante
    dd = 0
    dados +=dd
    
    #dados = 3506 # 
    
    with open(f'{pasta}//resolucao', 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dt = dado['dt']
        dbfile.close()
    
    resultado0 = analise(f'{pasta}\\dado-{0}')
    resultado1 = analise(f'{pasta}\\dado-{1}')
    
    env0 = resultado0['envelope']
    env1 = resultado1['envelope']
    t = resultado0['t']
    
    dD = np.abs(env0(t)-env1(t))
    
    #achar expoente de Lyapunov 
    dd = np.max(np.abs(resultado0['r0']-resultado1['r0']))
    serie = np.log(dD)
    
    def fit(x, a,b,xlin):
        return np.where( np.less(x,xlin), (a*x)+b, (a*xlin)+b)
    
    
    popt, pcov = curve_fit(fit, t, serie, p0=[10, 1e-8, 1.6])
  
    print(f'expoente de Lyapunov {popt[0]} Hz')
    print(f'delta fit {np.e**popt[1]} mm')
    print(f'delta {dd} mm')
    print(f't saturamento {popt[2]} s')
    
    plt.figure(dpi=300)
    plt.title(f'fit expoente de Lyapunov (simulação {pasta[-1]}) \n'+r'$\delta_0$ = ' + f'{dd:.0e} mm, dt = {dt:.1e}s')
    plt.semilogy(t,dD, label = 'dados')

    plt.semilogy(t, np.e**(fit(t, *popt)) , label=r'$\delta_0 e^{\lambda t}$, $\lambda$ = ' + f'{popt[0]:.2f} Hz,\n' + r'$\delta_0$ = ' + f'{np.e**popt[1]:.2e} mm,  t<{popt[2]:.1f} s')
    plt.legend()
    plt.ylabel(r'$\delta (t)$ [mm]')
    plt.xlabel(r't [s]')