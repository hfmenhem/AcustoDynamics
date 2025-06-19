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

def analise2(nome):
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
    
    Nsenos = 1
    Nvezes=6
    e=0.5#ordem de magnitude da diferença do pico para sua base (log(pico)-log(base)>e)
    m = 3 #amplitude de ordem de grandeza
    
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
        
        plt.figure(dpi=300)
        plt.title(f'{i}-1')
        plt.xlim((0,100))
        plt.semilogy(xf, yf, '.')
        plt.semilogy(xfpic, yfpic, '.')
        
        plt.figure(dpi=300)
        plt.title(f'resíduo-{i}-1')
        plt.plot(t, dados, '.')
        
        popts=[]
        j=1
        while(len(yfpic)>0 and j<=Nvezes):
            #Organizando picos
            indy = np.argsort(yfpic)
            xfpic = np.flip(xfpic[indy])
            yfpic = np.flip(yfpic[indy])
            
            
            try:
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
            except:
                break
            
            
      
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
            
            plt.figure(dpi=300)
            plt.title(f'{i}-{j}')
            plt.xlim((0,300))
            plt.semilogy(xf, yf, '.')
            plt.semilogy(xfpic, yfpic, '.')
            
            plt.figure(dpi=300)
            plt.title(f'resíduo-{i}-{j}')
            plt.plot(t, dados, '.')
            
        #print(f'razão desvio padrão / menor amplitude utilizada: {np.std(rs[i, :, 2]-senos(t, *popt))/abs(popt[Nsenos])}')
        
        #útima regressão para acomodar todos os parâmetros
        popts, pcov = curve_fit(senos, t, rs[i, :, 2], popts)
        
        npopts=len(popts[1:])//3

        freqs = popts[npopts+1:2*npopts+1]
        
   
        repetidos=np.tril(np.isclose(np.expand_dims(freqs,0), np.expand_dims(freqs,1), atol=5e-3), k=-1)
        if np.any(repetidos):
            argrep = np.where(repetidos)[0]
            
            popts=np.delete(popts,[ *(argrep+1), *argrep+npopts+1, *(argrep+2*npopts+1)])
            popts, pcov = curve_fit(senos, t, rs[i, :, 2], popts)
        
        plt.figure(dpi=300)
        plt.title(f'resíduo-{i}-final')
        plt.plot(t, rs[i, :, 2] - senos(t,*popts), '.')
        print(xf[1]-xf[0])
        
        npopts=len(popts[1:])//3
        ampsfinal.append(popts[1:npopts+1])
        freqsfinal.append(popts[npopts+1:2*npopts+1])
        fasesfinal.append(popts[2*npopts+1:3*npopts+1])
    
    resultado = {'r0': r0, 'freq': freqsfinal, 'amp': ampsfinal, 'fase': fasesfinal}
    return resultado

def analise1(nome):
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
    Nvezes=20
    e=0.5#ordem de magnitude da diferença do pico para sua base (log(pico)-log(base)>e)
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
        
        plt.figure(dpi=300)
        plt.title(f'{i}-1')
        plt.semilogy(xf, yf, '.')
        plt.semilogy(xfpic, yfpic, '.')
        
        plt.figure(dpi=300)
        plt.title(f'resíduo-{i}-1')
        plt.xlim((0,0.1))
        plt.plot(t, dados, '.')
        
        popts=[]
        j=1
        while(len(yfpic)>0 and j<=Nvezes):
            #Organizando picos
            indy = np.argsort(yfpic)
            xfpic = np.flip(xfpic[indy])
            yfpic = np.flip(yfpic[indy])
            
            
            try:
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
            except:
                break
            
            
      
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
            
            plt.figure(dpi=300)
            plt.title(f'{i}-{j}')
            plt.xlim((0,300))
            plt.semilogy(xf, yf, '.')
            plt.semilogy(xfpic, yfpic, '.')
            
            plt.figure(dpi=300)
            plt.title(f'resíduo-{i}-{j}')
            plt.xlim((0,0.1))
            plt.plot(t, dados, '.')
            
        #print(f'razão desvio padrão / menor amplitude utilizada: {np.std(rs[i, :, 2]-senos(t, *popt))/abs(popt[Nsenos])}')
        
       
        plt.figure(dpi=300)
        plt.title(f'resíduo-{i}-final')
        plt.xlim((0,0.1))
        plt.plot(t, rs[i, :, 2] - senos(t,*popts), '.')
        print(xf[1]-xf[0])
        
        npopts=len(popts[1:])//3
        ampsfinal.append(popts[1:npopts+1])
        freqsfinal.append(popts[npopts+1:2*npopts+1])
        fasesfinal.append(popts[2*npopts+1:3*npopts+1])
    
    resultado = {'r0': r0, 'freq': freqsfinal, 'amp': ampsfinal, 'fase': fasesfinal}
    return resultado



if __name__ == '__main__':
    
    pasta='Sim2'
    nome = 'Sim2'
    #dados = 114 # 18
    #dados = 140 # 47
    dados = 0 # 0
    #dados = 210 #Exemplo de pequenas oscilações
    
    caminho = f'{pasta}\\{nome}-dado-{dados}'
    
    with open(caminho, 'rb') as dbfile:
        dado = pickle.load(dbfile)
        dbfile.close()

    print('r0')
    print(dado['r0'][:, 2])
    
    resultado = analise1(caminho)
    
    
    
    for a, f, i in zip(resultado['amp'],resultado['freq'], range(2)):
        a = np.where(np.isnan(a),np.full(np.shape(a),0), a)
        arg = np.argsort(np.abs(a), axis=0)
       
        
        f = np.take_along_axis(np.array(f), arg, axis=0)
        a = np.take_along_axis(np.array(a), arg, axis=0)
        
        
        print(i)
        print('f')
        print(f)
        print('a')
        print(np.abs(a))


