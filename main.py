import numpy as np

class Simulador:
    
    def __init__(self, f1, f2, f, c, k, a, m, rho, v0, h):   
        self.f1 = f1
        self.f2 = f2
        self.f = f
        self.c = c
        self.k = k
        self.a = a
        self.m = m
        self.rho = rho
        self.v0 = v0
        self.h = h
        self.e = 1
        self.limDV = v0*1e-9 #Limite de varição de velocidade durante uma colisão. Caso o valor seja menor que isso em módulo, a colisão é recalculada considerando e=0
    
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
        
        coef1 = np.transpose(-self.f1*(self.a**3)*(self.k**2)/(3))
        coef2 = (-1j)*np.transpose(self.f2*(self.a**3))/(2)
        # tt= np.einsum('ijk,ijk->ij', gpin, R) = produto escalar element-wise do útlimo eixo
        
        part11 = np.expand_dims((3-(3j*self.k*Rn)-((self.k*Rn)**2))/(Rn**2), (2,3))*RxR 
        part12 = np.expand_dims(1j*(1j+(self.k*Rn)), (2,3))*I
        part1 = np.expand_dims(coef1*pin*(np.e**(1j*self.k*Rn))/(Rn**3), (2,3))*( part11 + part12)
        
        part211 = np.expand_dims((15J+(15*self.k*Rn) - (((self.k*Rn)**2)*(6j+(self.k*Rn))))/(Rn**2), (2,3))*RxR 
        part212 =  np.expand_dims((-3j-(3*self.k*Rn)-(1j*((self.k*Rn)**2))), (2,3))*I
        part21 = np.expand_dims(np.einsum('ijk,ijk->ij', gpin, R), (2,3)) * (part211 + part212)
        
        part22 =  np.expand_dims((-3+(3j*self.k*Rn)+((self.k*Rn)**2)), (2,3))*(Rgp+np.transpose(Rgp, axes=(0,1,3,2)))
        part2 = np.expand_dims(coef2*(np.e**(1j*self.k*Rn))/(Rn**5), (2,3))*( part21 + part22)
        
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
        #Essa matriz é simétrica
        MR = self.a + np.transpose(self.a, (1,0))
        #polinomio de 4 grau: cada elemento da lista vai ser multiplicado por [dt^4, dt^3, dt^2, dt, 1]
        #Cada elemento dessa lista é uma matriz simétrica
        indPolynomial = np.array([(np.linalg.norm(Ma0, axis=2)**2)/4, np.einsum('ijk,ijk->ij', Ma0, Mv0),(np.linalg.norm(Mv0, axis=2)**2) +np.einsum('ijk,ijk->ij', Ma0, Mr0),2*np.einsum('ijk,ijk->ij', Mv0, Mr0),(np.linalg.norm(Mr0, axis=2)**2) -(MR**2)])
        
        dtCol = np.apply_along_axis(self.MenorRaizReal, 0, indPolynomial)
        dtCol = dtCol+ np.triu(np.ones(len(dtCol))*np.inf)#Basicamente leva todos os valores da diagonal para cima como np.inf
        argAfastando = np.argwhere(np.logical_and(dtCol==0 ,np.einsum('ijk,ijk->ij', Mv0, Mr0)>0))#Onde as partículas estão encostadas mas se afastando
        dtCol[argAfastando]=np.inf
        return dtCol
    
    def MenorRaizReal(self,ind):
        v=np.roots(ind)
        v = v[np.isreal(v)]
        v = v[v>0]
        if len(v) ==0:
            v=[np.inf]
        
        return min(v) #Retorna a menor raiz real maior ou igual a 0
        
    def Simular(self, r0, v0, dt, tempo):
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
            
            F = self.FGorKov(Pt, GPt, HPt) #[uN]
            
            A = F/np.expand_dims(self.m, 2)
            
            dr = v*dt + A*(dt**2)/2
            dv = A*dt
            
            v = v+dv
            r = r+dr
            
            rs[:,t+1,:] = r[:,0,:]
            vs[:,t+1,:] = v[:,0,:]
        
        return rs, vs, np.array(range(frames))*dt
    
    def SimularComColisão(self, r0, v0, dt, tempo):
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
            
            F = self.FGorKov(Pt, GPt, HPt) #[uN]
            
            A = F/np.expand_dims(self.m, 2)
            
            #Testar se alguma esfera está encostada na outra, com velocidade radial nula
            
            CalculaColEnc=True
            while(CalculaColEnc):
                MRl = np.linalg.norm(MR, axis=2)- (self.a + np.transpose(self.a, (1,0)))
                MRl = MRl + np.triu(np.ones(np.shape(MR[:,:,0]))*np.inf) #Matriz triangular das distâncias entre partículas
                
                MV = v - np.transpose(v, (1,0,2))
                MVn = np.einsum('ijk,ijk->ij', MV, MR/np.linalg.norm(MR, axis=2, keepdims=True)) #Matriz triangular das Velocidades normais
                
                #Primiero calcular colisões
                indColisao = np.argwhere(np.logical_and(MRl==0 ,MVn<0))
                indEncostado = np.argwhere(np.logical_and(MRl==0 ,MVn==0))
                if np.size(indColisao) !=0:
                    Dv = self.calculaColisao(indColisao, r, v)
                    v = v+Dv
                elif np.size(indEncostado) !=0: #Quando Há colisão, MV muda, então temos que re-calcular isso, por isso pulamos isso e no próximo loop que fazemos 
                    DA, indEncostadoV0 = self.calculaEncostado(indEncostado, r, v, A)
                    A = A+DA
                    if np.all(DA==0):#Se não houve mais nenhuma modificação na aceleração, pode sair do loop
                        CalculaColEnc=False
                else: 
                    CalculaColEnc = False #Se não houve nenhuma colisão, então ele sai desse loop
            

            #Até aqui, A, v, r são as variáveis relevantes. Depois, é considerado A consntante e integrado para achar v' e r'. Porém, isso só acontece se não for ocorrer uma colisão
            
            dtcol = self.tempoParaColisao(r, v, A)
            dtValido = np.extract(np.logical_and(dtcol < dt, dtcol!=0),dtcol)
            if np.size(dtValido) !=0: #Ocorrerá uma colisão ou encostamento antes do próximo passo, então vamos simular até esse momento
                dtmincol = np.min(dtValido)
                dr = v*dtmincol + A*(dtmincol**2)/2
                dv = A*dtmincol
                
                v = v+dv
                r = r+dr
                tr = tr+dtmincol
                
                print('colidiu ou encostou!')
                print(tr)
            
            else:            
                dr = v*dt + (A*(dt**2)/2)
                dv = A*dt
                
                v = v+dv
                r = r+dr
                tr = tr+dt

            
            
            
            rs[:,t+1,:] = r[:,0,:]
            vs[:,t+1,:] = v[:,0,:]
            ts[t+1] = tr
            
        
        return rs, vs, ts
    
    def calculaColisao(self, indices, r, v):
        Dv = np.zeros(np.shape(v))
        for ind in indices:
            Dr = r[ind[0],0, :] - r[ind[1],0, :]
            drhat = Dr/np.linalg.norm(Dr)
            
            v0i = np.dot( v[ind[0], 0, :], drhat)
            v1i = np.dot( v[ind[1], 0, :], drhat)
            
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
        indV0 =[]
        for ind in indices:
            Dr = r[ind[0],0, :] - r[ind[1],0, :]
            drhat = Dr/np.linalg.norm(Dr)
            
            a0i = np.dot( a[ind[0], 0, :], drhat)  
            a1i = np.dot( a[ind[1], 0, :], drhat)  
            N = drhat*self.m[ind[0]]*self.m[ind[1]]*(a1i-a0i)/(self.m[ind[0]]+self.m[ind[1]])
            
            Da[ind[0], 0, :] = Da[ind[0], 0, :] + N/self.m[ind[0]]
            Da[ind[1], 0, :] = Da[ind[1], 0, :] - N/self.m[ind[1]]
            indV0.append(ind)
            
        return Da, indV0




























