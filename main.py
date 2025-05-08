import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import pickle 

class Simulador:
    
    def __init__(self, f1, f2, f, c, a, m, rho, v0, h, dinvis, plano = None, e=0.3):   
        self.f1 = f1
        self.f2 = f2
        self.f = f
        self.c = c
        self.k = 2*np.pi*f/c
        self.a = a
        self.m = m
        self.rho = rho
        self.v0 = v0
        self.h = h
        self.e = e
        self.limDV = v0*1e-9 #Limite de varição de velocidade durante uma colisão. Caso o valor seja menor que isso em módulo, a colisão é recalculada considerando e=0
        self.dinvis = dinvis
        
        #Plano:
        if plano is not None:
            self.Np = np.array(plano[0])
            self.Np = self.Np/np.linalg.norm(self.Np)
            self.Pp = np.array(plano[1])
            self.HaPlano = True
        else:
            self.HaPlano = False
    
    def agua(Npar):
        #unidades mm, s, g
        dicAgua = {'f1': np.array(Npar*[[0.623]]), 'f2':np.array(Npar*[[0.034]]), 'f': 10*(10**6), 'c':1480*(10**3), 'rho': 998*(10**(-6)), 'v0': (50*(10**3))/(998*(10**(-6))*1480*(10**3)), 'k':2*np.pi*(10*(10**6))/(1480*(10**3)), 'dimvis': 1.002}
        #v0 = (50*(10**3))/(c*rho) #Pressão = 50*(10**3) Pa = 50*(10**3) g/mm*s^2
        #viscosidade do ar [kg/m*s = g/mm*s] a 20°C tirada de https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm
        return dicAgua
    
    def ar(Npar):
        #unidades mm, s, g
        dicAr = {'f1': np.array(Npar*[[1.0]]), 'f2':np.array(Npar*[[1.0]]), 'f': 10*(10**3), 'c':340*(10**3), 'rho': 1.29*(10**(-6)), 'v0': 1e3, 'k':2*np.pi*(10*(10**3))/(340*(10**3)), 'dinvis': 1.849*(10**-5)}
        #v0 = = 1e3mm/s = (408)/(c*rho) #Pressão = 800 Pa = 800 g/mm*s^2
        #viscosidade do ar [kg/m*s = g/mm*s] a 25°C tirada de https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm
        return dicAr
    
    def PhiIn(self,r):
        return (self.v0/self.k)*np.sin(self.k*(r[:,:, 2]-self.h))
    
    def GradPhiIn(self,r):
        f = lambda a: [0,0, self.v0*np.cos(self.k*(a[2]-self.h))]
        return np.apply_along_axis(f, 2, r)
    
    def HPhiIn(self,r):
        f =lambda a: [[0,0,0], [0,0,0], [0,0, -self.k*self.v0*np.sin(self.k*(a[2]-self.h)) ]]
        return np.apply_along_axis(f, 2, r)
    
    def PhiSc(self, R, pin, gpin):
        Rn = np.linalg.norm(R, axis= 2)
        coef1 = -np.transpose(self.f1*(self.a**3)*(self.k**2))/(3)
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        #np.einsum('ijk,ijk->ij', gpin, R) #= produto escalar element-wise do útlimo eixo
        resul = (np.e**(1j*self.k*Rn))*( (coef1*pin/Rn) + (coef2*np.einsum('ijk,ijk->ij', gpin, R)*(1j+self.k*Rn)/(Rn**3)))
        np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return resul
    
    def GradPhiSc(self, R, pin, gpin):
        Rn = np.linalg.norm(R, axis= 2)
        coef1 = (-1j)*np.transpose(self.f1*(self.a**3)*(self.k**2))/(3)
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
        #
        part1=np.expand_dims(coef1*pin*(1j+(self.k*Rn))/(Rn**3), axis=2)*R
        part2=np.expand_dims(coef2*(1j+(self.k*Rn))/(Rn**3),axis=2)*gpin+np.expand_dims(coef2*((-3j -(3*self.k*Rn)+1j*((self.k*Rn)**2))/(Rn)**5)*np.einsum('ijk,ijk->ij', gpin, R), axis=2)*R
    
        resul = np.expand_dims(np.e**(1j*self.k*Rn), axis=2)*(part1 + part2)
        np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return resul
    
    def HPhiSc(self, R, pin, gpin):
        Rn = np.linalg.norm(R, axis= 2)
        RxR = np.expand_dims(R, 2) * np.expand_dims(R, 3)
        I = np.expand_dims(np.identity(3, dtype=float), (0,1))
        Rgp = np.expand_dims(R, 2) * np.expand_dims(gpin, 3)
        Rdotgp = np.expand_dims(np.einsum('ijk,ijk->ij', gpin, R), (2,3))
        
        coef1 = np.transpose(-self.f1*(self.a**3)*(self.k**2)/(3))
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
        
        part11 = np.expand_dims((3-(3j*self.k*Rn)-((self.k*Rn)**2))/(Rn**2), (2,3))*RxR 
        part12 = np.expand_dims(1j*(1j+(self.k*Rn)), (2,3))*I
        part1 = np.expand_dims(coef1*pin*(np.e**(1j*self.k*Rn))/(Rn**3), (2,3))*( part11 + part12)

        part21 = np.expand_dims((15j+(15*self.k*Rn)+(-6j*(self.k*Rn)**2)+(-1*(self.k*Rn)**3))/(Rn**2), (2,3))*RxR*Rdotgp
        part22 = np.expand_dims((-3j + (-3*(self.k*Rn))+(1j*(self.k*Rn)**2)), (2,3))*(Rgp+np.transpose(Rgp, axes=(0,1,3,2))+(Rdotgp*I))
        
        part2 = np.expand_dims(coef2*(np.e**(1j*self.k*Rn))/(Rn**5), (2,3))*(part21+part22)
       
        resul = part1+part2
        np.nan_to_num(resul, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return resul
    
    def FGorKov(self, Pt, GPt, HPt):
        part1 =np.expand_dims(self.f1*2/3, 2)*np.real(np.expand_dims(np.conjugate(Pt), 2)*GPt)
        part2 =np.expand_dims(-self.f2/(self.k**2), 2)*np.real( np.einsum('abj,abjk->abk', np.conjugate(GPt), HPt))
    
        f = np.expand_dims(-1*(self.a**3)*np.pi*self.rho*(self.k**2), 2) *(part1+part2)
        return f
    
    def tempoParaColisao(self, r0, v0, a0):
        #Essas matrizes não são tais que A = -At
        Mr0 = r0 - np.transpose(r0, (1,0,2))
        Mv0 = v0 - np.transpose(v0, (1,0,2))
        Ma0 = a0 - np.transpose(a0, (1,0,2))
            
        #Essa matriz é simétrica (se não houver plano)
        MR = self.a + np.transpose(self.a, (1,0))
        
        if self.HaPlano:
            RP = np.expand_dims(np.transpose(np.dot((r0-self.Pp) , self.Np)),2)*np.expand_dims(self.Np, (0,1))
            Mr0 = np.append(Mr0, RP, axis=0)
            VP = np.expand_dims(np.transpose(np.dot(v0 , self.Np)),2)*np.expand_dims(self.Np, (0,1))
            Mv0 = np.append(Mv0, VP, axis=0)
            AP =  np.expand_dims(np.transpose(np.dot(a0 , self.Np)),2)*np.expand_dims(self.Np, (0,1))
            Ma0 = np.append(Ma0, AP, axis=0)  
            MR = np.append(MR, np.transpose(self.a), axis=0)
        #polinomio de 4 grau: cada elemento da lista vai ser multiplicado por [dt^4, dt^3, dt^2, dt, 1]
        #Cada elemento dessa lista é uma matriz simétrica
        indPolynomial = np.array([(np.linalg.norm(Ma0, axis=2)**2)/4, np.einsum('ijk,ijk->ij', Ma0, Mv0),(np.linalg.norm(Mv0, axis=2)**2) +np.einsum('ijk,ijk->ij', Ma0, Mr0),2*np.einsum('ijk,ijk->ij', Mv0, Mr0),(np.linalg.norm(Mr0, axis=2)**2) -(MR**2)])
        
        dtCol = np.apply_along_axis(self.MenorRaizReal, 0, indPolynomial)#Retorna os menores tempos de colisão diferentes de 0
        dtCol = dtCol+ np.triu(np.full(np.shape(dtCol), np.inf))#Basicamente leva todos os valores da diagonal para cima como np.inf
        return dtCol
    
    def MenorRaizReal(self,ind):
        v=np.roots(ind)
        v = v[np.isreal(v)]
        v = v[v>0]
        if len(v) ==0:
            v=[np.inf]
        
        return min(v) #Retorna a menor raiz real maior ou igual a 0
        
    def Simular(self, r0, v0, dt, tempo, g=[0,0,0]):
        frames = int(tempo/dt)
        nPar = np.shape(v0)[0]
        r = r0
        v = v0

        rs = np.empty((nPar, frames, 3))
        vs = np.empty((nPar, frames, 3))

        rs[:,0,:] = r0[:,0,:]
        vs[:,0,:] = v0[:,0,:]

        #Inicio do loop
        for t in range(0, frames-1):
            MR = r - np.transpose(r, (1,0,2))
            Pin = self.PhiIn(r) #[mm^2/s]
            GPin = self.GradPhiIn(r)
            HPin = self.HPhiIn(r)
            
            Psc = self.PhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)) )
            GPsc = self.GradPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
            HPsc = self.HPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
            
            Pt  = np.sum(Psc , axis = 1,keepdims=True) + Pin
            GPt = np.sum(GPsc, axis = 1,keepdims=True) + GPin
            HPt = np.sum(HPsc, axis = 1,keepdims=True) + HPin
            
            Fac = self.FGorKov(Pt, GPt, HPt) #[uN]
            
            Far = -6*np.pi*self.dinvis*np.expand_dims(self.a, 2)*v
            A =( (Fac+Far)/np.expand_dims(self.m, 2)) + np.expand_dims(g, (0,1))
            
            dr = v*dt + A*(dt**2)/2
            dv = A*dt
            
            v = v+dv
            r = r+dr
            
            rs[:,t+1,:] = r[:,0,:]
            vs[:,t+1,:] = v[:,0,:]
        
        return rs, vs, np.array(range(frames))*dt
    
    def SimularComColisão(self, r0, v0, dt, tempo, g=[0,0,0]):
        frames = int(tempo/dt)
        nPar = np.shape(v0)[0]
        r = r0
        v = v0
        tr = 0

        rs = np.empty((nPar, frames, 3))
        vs = np.empty((nPar, frames, 3))
        ts = np.empty((frames))
        
        rs[:,0,:] = r0[:,0,:]
        vs[:,0,:] = v0[:,0,:]
        ts[0] = tr
        
        TColsisoes =[] #Para guardar o tempo em que as partículas se encostam
            
        #Inicio do loop
        t=0
        
        while tr<tempo:
            MR = r - np.transpose(r, (1,0,2))
            Pin = self.PhiIn(r) #[mm^2/s]
            GPin = self.GradPhiIn(r)
            HPin = self.HPhiIn(r)
            
            Psc = self.PhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)) )
            GPsc = self.GradPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
            HPsc = self.HPhiSc(MR, np.transpose(Pin), np.transpose(GPin, axes=(1,0,2)))
            
            Pt  = np.sum(Psc , axis = 1,keepdims=True) + Pin
            GPt = np.sum(GPsc, axis = 1,keepdims=True) + GPin
            HPt = np.sum(HPsc, axis = 1,keepdims=True) + HPin
            
            Fac = self.FGorKov(Pt, GPt, HPt) #[uN]
            
            Far = -6*np.pi*self.dinvis*np.expand_dims(self.a, 2)*v
            A =( (Fac+Far)/np.expand_dims(self.m, 2)) + np.expand_dims(g, (0,1))
            
            CalculaColEnc=True
            while(CalculaColEnc):
                MRl = np.linalg.norm(MR, axis=2)- (self.a + np.transpose(self.a, (1,0)))
                MRl = MRl + np.triu(np.ones(np.shape(MR[:,:,0]))*np.inf) #Matriz triangular das distâncias entre partículas
                
                MV = v - np.transpose(v, (1,0,2))
                MVn = np.einsum('ijk,ijk->ij', MV, MR/np.linalg.norm(MR, axis=2, keepdims=True)) + np.triu(np.ones(np.shape(MR[:,:,0]))*np.inf) #Matriz triangular das Velocidades normais
                
                if self.HaPlano:
                    RP = np.transpose(np.dot((r-self.Pp), self.Np) - self.a)
                    MRl = np.append(MRl, RP, axis=0)
                    VP = np.transpose(np.dot(v, self.Np))
                    MVn = np.append(MVn, VP, axis=0)
                    
                #Primiero calcular colisões
       
                indColisao = np.argwhere(np.logical_and(np.logical_and(np.logical_or(MRl<0, np.isclose(MRl, 0)),MVn<0), np.logical_not(np.isclose(MVn, 0)))) #A rigor, era pra ser MRL==0. Porém, para contornar problemas de erros numéricos, fazemos MRl<=0, porque existem casos que esse valor vai ser muito pequeno (em vez de 0, devido a erro numérico), porém negativo. Perceba que se MRl for muito pequeno, porém positivo, será feita mais uma passagem do loop até chegar no valor 0 ou muito pequeno negativo. Perceba também que nunca ocorrerá um valor de MRl de fato negativo (ou seja, negativo sem ser por erro numérico), visto que para isso uma esfera deveria atravessar outra, porém o programa impede isso de ocorrer quando MRl ==0 (caso preciso) ou MRl aprox 0 porém negativo (caso com um pequeno erro numérico)
                indEncostado = np.argwhere(np.logical_and(np.logical_or(MRl<0, np.isclose(MRl, 0)) , np.isclose(MVn, 0))) 

                if np.size(indColisao) !=0:
                    Dv = self.calculaColisao(indColisao, r, v)
                    v = v+Dv
                elif np.size(indEncostado) !=0: #Quando Há colisão, MV muda, então temos que re-calcular isso, por isso pulamos isso e no próximo loop que fazemos 
                    DA = self.calculaEncostado(indEncostado, r, v, A)
                    A = A+DA
                    if np.all(np.isclose(DA, 0)):#Se não houve mais nenhuma modificação na aceleração, pode sair do loop
                        CalculaColEnc=False
                else: 
                    CalculaColEnc = False #Se não houve nenhuma colisão, então ele sai desse loop
            

            #Até aqui, A, v, r são as variáveis relevantes. Depois, é considerado A consntante e integrado para achar v' e r'. Porém, isso só acontece se não for ocorrer uma colisão
            
            dtcol = self.tempoParaColisao(r, v, A)
            dtValido = np.extract(np.logical_and(dtcol < dt, dtcol >0),dtcol)
            
            MA = A - np.transpose(A, (1,0,2))
            MAn = np.einsum('ijk,ijk->ij', MA, MR/np.linalg.norm(MR, axis=2, keepdims=True))
            if self.HaPlano:
                AP = np.transpose(np.dot(A, self.Np))
                MAn = np.append(MAn, AP, axis=0)
            
            Dr = (MVn**2)/(-2*MAn)
            
            argerroR = np.argwhere( np.logical_and(np.logical_and(MAn<0, MVn>0), Dr<-1*MRl))#Isso significa que há duas esferas dentro uma da outra, se afastando, porém a velocidade de afastamento não é suficiente para se afastarem completamente
            if len(argerroR)!=0:
                dterroR = np.squeeze((-MVn/MAn)[argerroR[:,0],argerroR[:,1]])
                dterroR = np.extract(dterroR < dt,dterroR)
            else:
                dterroR=[]
            if np.size(dtValido) !=0 or np.size(dterroR)!=0: #Ocorrerá uma colisão ou encostamento antes do próximo passo, então vamos simular até esse momento
                dtmincol = np.min(np.concatenate((dtValido, dterroR)))
                dr = v*dtmincol + A*(dtmincol**2)/2
                dv = A*dtmincol
                
                v = v+dv
                r = r+dr
                tr = tr+dtmincol
                
                if np.size(dtValido) !=0:
                    print('colidiu ou encostou!')
                    print(tr)
                    TColsisoes.append(tr)
            
            else:            
                dr = v*dt + (A*(dt**2)/2)
                dv = A*dt
                
                v = v+dv
                r = r+dr
                tr = tr+dt

            if np.shape(rs)[1]==t+1:
                rs = np.append(rs, np.zeros((nPar, 100 , 3)), axis=1)
                vs = np.append(vs, np.zeros((nPar, 100 , 3)), axis=1)
                ts = np.append(ts, np.zeros(100))
            rs[:,t+1,:] = r[:,0,:]
            vs[:,t+1,:] = v[:,0,:]
            ts[t+1] = tr
            t+=1
            
        rs=rs[:, 0:t+1, :] #como logo em cima foi somado 1 a t, os dados estão entre 0:t+1, que originalmente era 0:t+2 (porque ele não inclui o último valor)
        vs=vs[:, 0:t+1, :]
        ts=ts[0:t+1]
        
        return rs, vs, ts, TColsisoes
    
    def calculaColisao(self, indices, r, v):
        Dv = np.zeros(np.shape(v))
        for ind in indices:
            if ind[0]>=np.shape(r)[0]:#está colidindo com um plano (pela contrução de MRl, sempre será o primeiro índice)
            
                v1i = np.dot( v[ind[1], 0, :], self.Np)
                Dv1 = self.Np*(1+self.e)*(-v1i)
                
                if np.linalg.norm(Dv1) < self.limDV  and self.e!=0:
                    Dv1 = self.Np*(-v1i)

                Dv[ind[1], 0, :] = Dv[ind[1], 0, :] + Dv1
            
            else:
                Dr = r[ind[0],0, :] - r[ind[1],0, :]
                                
                v0i = np.dot( v[ind[0], 0, :], Dr)/np.linalg.norm(Dr)
                v1i = np.dot( v[ind[1], 0, :], Dr)/np.linalg.norm(Dr)
                
                drhat = Dr/np.linalg.norm(Dr)
                
                Dv0 = drhat*(1+self.e)*self.m[ind[1]]*(v1i-v0i)/(self.m[ind[1]]+self.m[ind[0]])
                Dv1 = drhat*(1+self.e)*self.m[ind[0]]*(v0i-v1i)/(self.m[ind[1]]+self.m[ind[0]])
                
                if( np.linalg.norm(Dv0) < self.limDV) and( np.linalg.norm(Dv1) < self.limDV ) and self.e!=0:
                    Dv0 = drhat*self.m[ind[1]]*(v1i-v0i)/(self.m[ind[1]]+self.m[ind[0]])
                    Dv1 = drhat*self.m[ind[0]]*(v0i-v1i)/(self.m[ind[1]]+self.m[ind[0]])
                
                Dv[ind[0], 0, :] = Dv[ind[0], 0, :] + Dv0
                Dv[ind[1], 0, :] = Dv[ind[1], 0, :] + Dv1
            
        return Dv
    
    def calculaEncostado(self, indices, r, v, a):
        Da = np.zeros(np.shape(a))
        
        if self.HaPlano:
            rlinha = np.append(r, np.full((1,1,3), np.nan), axis=0) #Colocamos um valor qualquer para a posição do plano, depois substituimos diretamente o valor de drhat como a normal
            Dr = rlinha[indices[:, 1],0, :] - rlinha[indices[:, 0],0, :]
            drhat = Dr/np.linalg.norm(Dr, axis=1, keepdims=True)
            
            drhat=np.where(np.expand_dims((indices[:,0]>=np.shape(r)[0]), 1), np.expand_dims(self.Np, 0), drhat)
            
            alinha =np.append(a, np.zeros((1,1,3)), axis=0)#o plano possui aceleração 0
            Da = alinha[indices[:, 1],0, :] - alinha[indices[:, 0],0, :]
        else:
            Dr = r[indices[:, 1],0, :] - r[indices[:, 0],0, :]
            drhat = Dr/np.linalg.norm(Dr, axis=1, keepdims=True)
            
            Da = a[indices[:, 1],0, :] - a[indices[:, 0],0, :]
        
        
        Da = np.einsum('ij,ij->i', Da, drhat)
        
        M = np.einsum('ijk,ijk->ij', np.expand_dims(drhat, 1), np.expand_dims(drhat, 0)) #= produto escalar element-wise do útlimo eixo
        
        O1 = np.zeros((len(indices),len(indices)))
        O2 = np.zeros((len(indices),len(indices)))
        
        O1[np.expand_dims(indices[:,0] , 1)==np.expand_dims(indices[:,0] , 0) ]+= -1
        O1[np.expand_dims(indices[:,0] , 1)==np.expand_dims(indices[:,1] , 0) ]+= 1
        
        O2[np.expand_dims(indices[:,1] , 1)==np.expand_dims(indices[:,0] , 0) ]+= -1
        O2[np.expand_dims(indices[:,1] , 1)==np.expand_dims(indices[:,1] , 0) ]+= 1
        
        if self.HaPlano:
            mlinha = np.append(self.m, np.full((1,1), np.inf), axis=0)
            O1 = O1/mlinha[indices[:,0]]
            O2 = O2/mlinha[indices[:,1]]
        else:
            O1 = O1/self.m[indices[:,0]]
            O2 = O2/self.m[indices[:,1]]
       
        
        Ns = np.linalg.solve(M*(O1-O2), Da)
                
        while np.any(Ns<0): #Caso tenha dado alguma normal negativa, repete o processo
            inderro = np.argwhere(Ns<0)
            M = np.delete(M, inderro, 0)
            M = np.delete(M, inderro, 1)
            
            O1 = np.delete(O1, inderro, 0)
            O1 = np.delete(O1, inderro, 1)
            
            O2 = np.delete(O2, inderro, 0)
            O2 = np.delete(O2, inderro, 1)
            
            Da = np.delete(Da, inderro)
            drhat = np.delete(drhat, inderro,0)
            indices =np.delete(indices, inderro,0)
            Ns = np.linalg.solve(M*(O1-O2), Da)
             
        Nvec = (np.expand_dims(Ns, 1)*drhat)

        Mind = np.zeros((len(r)+1,len(indices)))#Adicina=se um elemento a mais na linha das partículas para comportar o plano
        Mind[indices[:,0],range(len(indices[:,1]))] += -1
        Mind[indices[:,1],range(len(indices[:,1]))] += +1
        Mind = np.delete(Mind, len(r), axis=0) #No final é apagada a linha do plano
        
        DA = np.einsum('ij,ki->kj', Nvec, Mind)/self.m
            
        return np.expand_dims(DA, 1)
    
    def saveSimulacao(self, rs, vs, t, nome):
        if self.HaPlano:
            salvar = {'rs': rs, 'vs': vs, 't': t, 'f1': self.f1, 'f2': self.f2, 'f': self.f, 'c': self.c, 'a': self.a, 'm': self.m, 'rho': self.rho, 'v0': self.v0, 'Np': self.Np, 'Pp': self.Pp}
        else:
            salvar = {'rs': rs, 'vs': vs, 't': t, 'f1': self.f1, 'f2': self.f2, 'f': self.f, 'c': self.c, 'a': self.a, 'm': self.m, 'rho': self.rho, 'v0': self.v0}
        

        with open(nome+'.pkl', 'wb') as file: 
            pickle.dump(salvar, file)   
            
    def video(nome, FPS = 20, Tmult = 1):
        with open(nome+'.pkl', 'rb') as file:   
          salvo = pickle.load(file) 
          
        rs = salvo['rs']
        t = salvo['t']
        a = salvo['a']
        
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(projection='3d', azim=-80, elev=10,computed_zorder=False)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        #tempos dos frames
        tfps = np.linspace(0, np.max(t), round(np.max(t)*FPS*Tmult + 1))
        indTfps = np.argmin((np.abs(t - np.expand_dims(tfps, 1))),axis= 1)
     
        
        # esfera unitária
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        esfunit = np.array([np.outer(np.cos(u), np.sin(v)),np.outer(np.sin(u), np.sin(v)), np.outer(np.ones(np.size(u)), np.cos(v))])        
        
        rlim = np.array([np.min(rs, axis=(0,1)) - 2*np.min(a), np.max(rs, axis=(0,1)) + 2*np.max(a)])
        ax.set(xlim=rlim[:, 0], ylim=rlim[:, 1], zlim=rlim[:, 2] )
        ax.grid(False)
        ax.set_aspect('equal')    
        ax.set_axis_off()
        
        if 'Np' in salvo:
            Np = salvo['Np']
            Pp = salvo['Pp']

            
            if np.linalg.norm(np.cross(Np, [1,0,0]))==0:
                v1 = np.array([0,0,1])
            else:
                v1 = np.cross(Np, [1,0,0])/np.linalg.norm((np.cross(Np, [1,0,0])))
                      
            v2 = np.cross(v1, Np)/np.linalg.norm(np.cross(v1, Np))
        
            Ptsp = np.array([[v2+v1,v2-v1], [-v2+v1, -v2-v1] ])*(np.linalg.norm( rlim[1,:]-rlim[0,:])/3) + Pp
            
            ax.plot_surface(*np.transpose(Ptsp, (2,0,1)), color='xkcd:slate',zorder =0)
      
        ax.quiver(*np.transpose(3*[rlim[0, :]]), [0, 0, 1], [0, 1, 0], [1, 0, 0], length= (np.mean(rlim[1,:]-rlim[0,:])/5),arrow_length_ratio=0.2, linewidths=1, colors='k' )
        ax.text(*rlim[0, :], 'x', 'x', verticalalignment='top', horizontalalignment ='left')
        
        ax.scatter([0], [0], [0], color='k', s=10)
        
        
       
            
            
            
        artEsfera=[]
        for i, ai in enumerate(a):
            artEsfera.append(ax.plot_surface(*(ai*esfunit)+np.expand_dims(rs[i, 0, :], (1,2)), color='xkcd:teal'))
        #plt.show()
        
        maxframe=max(indTfps)
        def update(frame):
            for i, ai in enumerate(a):
                artEsfera[i].remove()
                artEsfera[i]=ax.plot_surface(*(ai*esfunit)+np.expand_dims(rs[i, frame, :], (1,2)), color='xkcd:teal')
            print(f'{ 100*frame/maxframe:.2f}% feito')
            return (artEsfera[0])
        ani = animation.FuncAnimation(fig=fig, func=update, frames=indTfps, interval=1000/FPS)
        ani.save(nome+'.gif')

    def graficos(rs, vs, t, TColsisoes, a):
        cmap = mpl.colormaps['viridis']
        Npar = len(a)
        
        CorTempo = cmap(np.linspace(0, 1, len(t))) #Mapa de cor para indicar o tempo da simulação

        #Achar tempos para plotar circulos
        ti = np.linspace(0, np.max(t), 5)
        indTi = np.argmin((np.abs(t - np.expand_dims(ti, 1))),axis= 1)

        #Achando indices da colisão
        indCol = np.any(np.equal(np.expand_dims(t, 1), np.expand_dims(TColsisoes,0)), axis=1)
        indtempocol = np.arange(0, len(indCol),1)[indCol]
        rColisao  = rs[:, indCol,:]

        #achando todos os pontos em que queremos plotar as esferas como circulos
        indplotR = np.concatenate((indTi, indtempocol))
        rplotR = rs[:, indplotR, :]


        #Plot do plano xz
        plt.figure(dpi=300)
        plt.axes().set_aspect('equal')
        #Definição do circulo unitário
        ang=np.linspace(0, 2*np.pi, 50)
        circ = np.transpose([np.sin(ang), np.zeros(len(ang)), np.cos(ang)])

        for i in range(Npar):
            plt.plot(rs[i, :, 0], rs[i, :, 2], linestyle='', marker='.',markersize=2)
            plt.plot(rColisao[i, :, 0], rColisao[i, :, 2], linestyle='', marker='.',markersize=2)
            
            cirplotR=rplotR[i, :,:] + np.expand_dims(a[i]*circ, 1)
            for c, indc in enumerate(indplotR):
                plt.plot(cirplotR[:,c, 0], cirplotR[:, c,2], linestyle='', marker='.',markersize=1, color=CorTempo[indc])


        plt.xlabel("x [mm]")
        plt.ylabel("z [mm]")

        plt.show()

        #x x t
        plt.figure(dpi=300)
        for i in range(Npar):
            plt.plot(t, rs[i, :, 0], linestyle='', marker='.',markersize=2)
            plt.plot(TColsisoes, rColisao[i, :, 0], linestyle='', marker='.',markersize=2)

        plt.ylabel("x [mm]")
        plt.xlabel("t [s]")

        plt.show()

        #z x t
        plt.figure(dpi=300)
        for i in range(Npar):
            plt.plot(t, rs[i, :, 2], linestyle='', marker='.',markersize=2)
            plt.plot(TColsisoes, rColisao[i, :, 2], linestyle='', marker='.',markersize=2)

        plt.ylabel("z [mm]")
        plt.xlabel("t [s]")

        plt.show()
























