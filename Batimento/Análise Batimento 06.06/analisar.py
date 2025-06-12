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
import pickle
import concurrent.futures

def senos(t, med, *kwarg):
    t = np.expand_dims(t, 0)
    N = len(kwarg)//3
    amp = np.expand_dims(np.array(kwarg[0:N]),1)
    freq = np.expand_dims(np.array(kwarg[N:2*N]),1)
    fase = np.expand_dims(np.array(kwarg[2*N:3*N]),1)
    return med+np.sum((amp*np.sin((2*np.pi*freq*t)+fase)), axis=0)


def analise(nome):
    with open(nome, 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dbfile.close()
        
    t = dado['t']
    rs = dado['rs']
    r0 = dado['r0']
    
    Npar=2
    dt = t[1]-t[0]
    
    ampsfinal=[]
    freqsfinal=[]
    fasesfinal=[]
    
    Nsenos = 2
    Nvezes=np.inf
    e=1.5#ordem de magnitude da diferença do pico para sua base (log(pico)-log(base)>e)
    m = 2 #amplitude de ordem de grandeza
    
    for i in range(Npar):
        N=len(rs[i, :, 2])
        #yf= (2.0/N)*np.abs(fft(rs[i, :, 2])[0:N//2])
        dados=rs[i, :, 2]
        xf=fftfreq(N, dt)[:N//2]
        
        #DFT
        yf= (2.0/N)*np.abs(fft(dados)[0:N//2])
        
        #Achando picos
        indpic = sig.find_peaks(yf)[0]
        
        picosProeminentes=sig.peak_prominences(yf,indpic)[0]/yf[indpic] >(1/(1+(10**(-e))))
    
        xfpic = (xf[indpic])[picosProeminentes]
        yfpic = (yf[indpic])[picosProeminentes]
        
        popts=[]
        j=1
        while(len(yfpic)>0 and j<=Nvezes):
            #Organizando picos
            indy = np.argsort(yfpic)
            xfpic = np.flip(xfpic[indy])
            yfpic = np.flip(yfpic[indy])
            
            #Ajuste
            if(len(yfpic)<Nsenos):
                p0=[np.mean(dados), *yfpic, *xfpic, *np.zeros(len(yfpic))]
                popt, pcov = curve_fit(senos, t, dados, p0)
            else:
                p0=[np.mean(dados), *yfpic[:Nsenos], *(xfpic[:Nsenos]), *np.zeros(Nsenos)]
                popt, pcov = curve_fit(senos, t, dados, p0)
            
            #Guardando dados da regressão
            if len(popts)>0:
                meds = popts[0]
            else:
                meds=0
            npopts=len(popts[1:])//3
            amps =popts[1:npopts+1]
            freqs = popts[npopts+1:2*npopts+1]
            fases = popts[2*npopts+1:3*npopts+1]
            
            med = popt[0]
            npopt=len(popt[1:])//3
            amp =popt[1:npopt+1]
            freq = popt[npopt+1:2*npopt+1]
            fase = popt[2*npopt+1:3*npopt+1]
            
            popts=[(meds+med), *amps,*amp, *freqs, *freq, *fases, *fase]
      
            #Recalculando resíduos
            dados = dados-senos(t, *popt)
            
            #DFT
            yf= (2.0/N)*np.abs(fft(dados)[0:N//2]) 
            
            #Achando picos
            indpic = sig.find_peaks(yf)[0]
            picosProeminentes=sig.peak_prominences(yf,indpic)[0]/yf[indpic] >(1/(1+(10**(-e))))
    
            xfpic = (xf[indpic])[picosProeminentes]
            yfpic = (yf[indpic])[picosProeminentes]
            
            limiteamp = yfpic>(np.max(np.abs([*amp, *amps]))*(10**(-m)))
            xfpic = xfpic[limiteamp]
            yfpic = yfpic[limiteamp]
            
            j+=1

        print(f'razão desvio padrão / menor amplitude utilizada: {np.std(rs[i, :, 2]-senos(t, *popt))/abs(popt[Nsenos])}')

        npopts=len(popts[1:])//3
        ampsfinal.append(popts[1:npopts+1])
        freqsfinal.append(popts[npopts+1:2*npopts+1])
        fasesfinal.append(popts[2*npopts+1:3*npopts+1])
    
    resultado = {'r0': r0, 'freq': freqsfinal, 'amp': ampsfinal, 'fase': fasesfinal}
    return resultado



if __name__ == '__main__':
    
    pasta='Sim1-v2'
    dados =[]
    for x in os.listdir(pasta):
        if 'dado' in x:
            dados.append(f'{pasta}\\{x}')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(analise, dados)
    
    r0s=[]
    amps=[]
    freqs=[]
    fases=[]
    for r in result:
        #Iguala o número de elementos dos arrays de cada esfera
        dif = len(r['amp'][0])-len(r['amp'][1])
        add = abs(dif)*[np.nan]
        if dif>0:
           r['amp'][1] = np.concatenate((r['amp'][1], add))
           r['freq'][1] = np.concatenate((r['freq'][1], add))
           r['fase'][1] = np.concatenate((r['fase'][1], add))
        elif dif<0:
            r['amp'][0] = np.concatenate((r['amp'][0], add))
            r['freq'][0] = np.concatenate((r['freq'][0], add))
            r['fase'][0] = np.concatenate((r['fase'][0], add))
        

        r0s.append(np.array(r['r0']))
        amps.append(np.array(r['amp']))
        freqs.append(np.array(r['freq']))
        fases.append(np.array(r['fase']))
    
    resultadoFinal ={'r0': r0s, 'freq': freqs, 'amp': amps, 'fase': fases}
    
    with open(f'{pasta}\\resultado-final', 'wb') as dbfile:
        pickle.dump(resultadoFinal, dbfile)
        dbfile.close()
    
    



