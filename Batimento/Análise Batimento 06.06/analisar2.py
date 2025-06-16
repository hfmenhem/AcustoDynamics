import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir) #Para poder importar o main, que está em uma pasta anterior

import numpy as np
from scipy.fft import fft, fftfreq
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
import pickle
import concurrent.futures
import time
start_time = time.time()

def analise(nome):
    with open(nome, 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dbfile.close()
        
    t = dado['t']
    rs = dado['rs']
    r0 = dado['r0']
    
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
        intbasico = make_interp_spline(t, rs[i, :, 2])
        tl=np.linspace(0, t[-1], 100*len(t))
        rsinterpolado = intbasico(tl)
        
        indpicM = sig.find_peaks(rsinterpolado)[0]

        intf = make_interp_spline(tl[indpicM], rsinterpolado[indpicM])

        indpicMM = sig.find_peaks(intf(t))[0]
        indpicMm = sig.find_peaks(-1*intf(t))[0]
        
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
        
    resultado = {'r0': r0, 'Pbat': Pbat, 'sPbat': sPbat, 'Posc': Posc, 'sPosc': sPosc, "Amax": Amax, "sAmax": sAmax, "Amin": Amin, "sAmin": sAmin}
    return resultado


if __name__ == '__main__':
    
    pasta='Sim6'
    dados =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            dados.append(f'{pasta}\\{x}')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(analise, dados)
    
    r0s=[]
    Pbats=[]
    sPbats=[]
    Poscs=[]
    sPoscs=[]
    Amaxs = []
    sAmaxs = []
    Amins = []
    sAmins = []
    for r in result:
        r0s.append(np.array(r['r0']))
        Pbats.append(np.array(r['Pbat']))
        sPbats.append(np.array(r['sPbat']))
        Poscs.append(np.array(r['Posc']))
        sPoscs.append(np.array(r['sPosc']))
        Amaxs.append(np.array(r['Amax']))
        sAmaxs.append(np.array(r['sAmax']))
        Amins.append(np.array(r['Amin']))
        sAmins.append(np.array(r['sAmin']))
        
        
    
    resultadoFinal ={'r0': r0s, 'Pbat': Pbats, 'sPbat': sPbats, 'Posc': Poscs, 'sPosc': sPoscs, "Amax": Amaxs, "sAmax": sAmaxs, "Amin": Amins, "sAmin": sAmins}
    
    with open(f'{pasta}\\resultado-final2', 'wb') as dbfile:
        pickle.dump(resultadoFinal, dbfile)
        dbfile.close()
        
    print(f'O código demorou {(time.time() - start_time):.1f} s')
    
    



