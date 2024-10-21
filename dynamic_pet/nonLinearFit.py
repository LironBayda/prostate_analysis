from os import listdir

from os.path import join

# import cupy as cp

import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt

from scipy import optimize

from os import listdir

from os.path import join

# import cupy as cp

import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt

from scipy import optimize


class nonLinearFit:

    def __init__(self):

        self.end_slice = 0

        self.start_slice = 0

    def save_2plot(self, path, x1, y1, x2, y2, title, x_label, y_label):

        print('plot')

        fig, ax = plt.subplots()

        ax.plot(x1, y1, 'ro', x2, y2)

        ax.set_title(title)

        ax.set_xlabel(x_label)

        ax.set_ylabel(y_label)

        fig.savefig(join(path, title + ".png"))

    
from os import listdir

from os.path import join

# import cupy as cp

import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt

from scipy import optimize

from os import listdir

from os.path import join

# import cupy as cp

import numpy as np

import nibabel as nib

import matplotlib.pyplot as plt

from scipy import optimize


class nonLinearFit:

    def __init__(self):

        self.end_slice = 0

        self.start_slice = 0

    def save_2plot(self, path, x1, y1, x2, y2, title, x_label, y_label):

        print('plot')

        fig, ax = plt.subplots()

        ax.plot(x1, y1, 'ro', x2, y2)

        ax.set_title(title)

        ax.set_xlabel(x_label)

        ax.set_ylabel(y_label)

        fig.savefig(join(path, title + ".png"))
        plt.close('all')


    def convolve_2cm(self, data, K1, K2, K3, K4,V):
        time = np.asarray(data[0])

        blood = np.asarray(data[1])
        cp=blood
        delta=np.sqrt(np.power(K2+K3+K4,2)-4*K2*K4)
        theta1=(K2+K3+K4+delta)/2
        theta2=(K2+K3+K4-delta)/2
        phi1=(K1*(theta1-K3-K4))/delta
        phi2=(K1*(theta2-K3-K4))/(-delta)
        ct=[]
        for i,t in enumerate(time[:-1]):
            H1=phi1*np.exp(-theta1*(t-time[0:i+1]))
            H2=phi2*np.exp(-theta2*(t-time[0:i+1]))
            ct.append((1-V)*np.sum(H*cp[0:i+1]*np.diff(np.insert(time[0:i+1],0,0)))+V*blood[i])
        return ct



    def convolve_1cm(self, data, K1, K2,V):
        time = np.asarray(data[0])

        blood = np.asarray(data[1])
        cp=blood
        theta=K2
        phi=K1
        ct=[]
        for i,t in enumerate(time):
            H=phi*np.exp(-theta*(t-time[0:i+1]))
            ct.append((1-0)*np.sum(H*cp[0:i+1]*np.diff(np.insert(time[0:i+1],0,0)))+0*blood[i])

##            ct.append((1-0)*np.trapz(H*cp[0:i+1],x=time[0:i+1])+0*blood[i])
        return ct
    def one_cm_fit_p(self, path, tac, c_p, time, slices_duration,V, title,sigma,i):
        self.V=V

        popt, pcov = optimize.curve_fit(self.convolve_1cm, [time, c_p, slices_duration], tac,p0=[0.2,0.1,0.01])

        #popt, pcov = optimize.curve_fit(self.convolve_1cm, [time, c_p, slices_duration], tac, maxfev=10000000)

        if i%100000==0:    
             self.save_2plot(path, time, tac,
                           time, self.convolve_1cm([time, c_p, slices_duration], popt[0], popt[1], popt[2]),
                            title, 'time', 'con')

        return popt[0], popt[1], popt[2]

    def one_cm_fit(self, path, tac, c_p, time, slices_duration,V, title,sigma,i):
        #self.V=V

       # popt, pcov = optimize.curve_fit(self.convolve_1cm, [time, c_p, slices_duration], tac, bounds=[0,1],p0=[0.2,0.1,0.01])
        try:
            popt, pcov = optimize.curve_fit(self.convolve_1cm, [time, c_p, slices_duration], tac, maxfev=100000)
        except:
            popt=[0,0,0]

        if i%100000==0:    
             self.save_2plot(path, time, tac,
                           time, self.convolve_1cm([time, c_p, slices_duration], popt[0], popt[1], popt[2]),
                            title, 'time', 'con')
        if popt[0]<0 or popt[1]<0 or popt[0]>1 or popt[1]>1 :
            popt=[0,0,0]
        

        return popt[0], popt[1], popt[2]



    def convolve_2cm_for_minimaze_trapz(self,X,*params):
        K1, K2, K3, K4,V = X
        blood,tac,one_div_sigma,time,dt=params
        tac[tac<0]=0
        cp=blood
        delta=np.sqrt(np.power(K2+K3+K4,2)-4*K2*K4)
        theta1=(K2+K3+K4+delta)/2
        theta2=(K2+K3+K4-delta)/2
        phi1=(K1*(theta1-K3-K4))/delta
        phi2=(K1*(theta2-K3-K4))/(-delta)
        ct=[]
        for i,t in enumerate(time):
            H1=phi1*np.exp(-theta1*(t-time[0:i+1]))
            H2=phi2*np.exp(-theta2*(t-time[0:i+1]))
            ct.append((1-0)*np.trapz(H1*cp[0:i+1],x=time[0:i+1])+V*blood[i]+(1-0)*np.trapz(H2*cp[0:i+1],x=time[0:i+1])+V*blood[i])
        return ct

        return np.sum(np.power(((1-V)*ct+V*blood.T)-tac,2))  

    def convolve_2cm_for_minimaze_3k(self,X,*params):
        K1, K2, K3, K4,V = X
        
        blood, time,tac,W,t_minus_time,dt=params
        tac[tac<0]=0
 

        cp=blood[1:,:]
        delta=np.sqrt(np.power(K2+K3,2))
        theta1=(K2+K3+delta)/2
        theta2=(K2+K3-delta)/2
        phi1=(K1*(theta1-K3))/delta
        phi2=(K1*(theta2-K3))/(-delta)
        H1=phi1*np.exp(-theta1*t_minus_time)
        H2=phi2*np.exp(-theta2*t_minus_time)
        H1=H1[1:,:]
        H2=H2[1:,:]

        blood=blood[1:,:]
        
        ct=np.sum(H1[:,1:]*cp[:,1:]*dt)+\
           np.sum(H2[:,1:]*cp[:,1:]*dt)
        return sum(W[1:]*np.power((1-V)*ct+V*blood[-1,1:].T-tac[1:],2))        
            


    def convolve_1cm_for_minimaze_trapz(self,X,*params):
        K1, K2,V = X        
        blood,tac,one_div_sigma,time,dt=params


        cp=blood
        theta=K2
        phi=K1
        ct=[]
        for i,t in enumerate(time):
            H=phi*np.exp(-theta*(t-time[0:i+1]))
            ct.append(np.trapz(H*cp[0:i+1],x=time[0:i+1]))

        ct=np.asarray(ct)
        return np.sum(one_div_sigma*np.power(((1-V)*ct+V*blood)-tac,2))


           
            
##    def convolve_1cm_for_minimaze(self,X,*paramss):
##        K1,K2,V = X
##        time,blood,tac,W=params
##        Bj = np.asarray([])
##      
##        
##        for i in range(24):
##         t=time[i]
##
##         i = i + 1
##         tua = time[0:i]
##
##         c_p = blood[0:i]
##
##         dtua = np.diff(np.insert(tua, 0, 0))
##
##         y = K1*np.exp(-(K2) * (t - tua)) * c_p
##
##         Bj = np.append(Bj,np.sum(dtua*y))
##        return   np.sum(W*np.power(((1-V)* Bj+V*blood) -tac,2))

    
    def convert_vector_to_matrix(self,vector):
        matrix=np.zeros([len(vector),len(vector)])
        for i,x in enumerate(vector):
            matrix[i:,i]=x
        return matrix
            

        
    def sa_optimaze(self, path, tac, blood, time, slices_duration,Vd, title,W,i):
        self.Vd=Vd
        x0=[0.7,0.2,0.5,0.05,0.01]
       
        #tac=np.insert(tac,0,0)
        #time=np.insert(time,0,0)
        #blood=np.insert(blood,0,0)
        dt=np.diff(time)

        dt=self.convert_vector_to_matrix(dt)
        blood_opt=self.convert_vector_to_matrix(blood)
        t_minus_time=np.zeros(blood_opt.shape)
        for i,t in enumerate(time):
            t_minus_time[i,0:i+1]=t-time[0:i+1]
        time=self.convert_vector_to_matrix(time)

        params= blood_opt, np.asarray(time),tac,W,t_minus_time,dt
        lw = [0,0,0,0,0]
        up = [1]*5

##        ret =optimize.dual_annealing(self.convolve_2cm_for_minimaze_3k, bounds=list(zip(lw, up)),args=params,no_local_search=True)
        ret = optimize.minimize(self.convolve_2cm_for_minimaze_3k,x0,args=params, method='Powell', bounds=list(zip(lw, up)),)
##                                         options={ 'xtol': 1e-4 ,'ftol':  1e-2,  'disp': False, 'return_all': False})

        if ret.success:
##            sol_tac=self.convolve_2cm([time, blood, slices_duration], ret.x[0], ret.x[1], ret.x[2], ret.x[3],ret.x[4])
##            self.save_2plot(path, time, tac,time, sol_tac,title, 'time', 'con') 
            return  ret.x[0], ret.x[1], ret.x[2], ret.x[3],ret.x[4]
          

    
       
         #ret = optimize.minimize(self.convolve_2cm_for_minimaze,x0,args=params, method='Powell', bounds=list(zip(lw, up)),
         #                        options={ 'xtol': 1e-2, 'ftol': 1e-2,  'disp': False, 'return_all': False})
         
     
       
      
        return 0,0,0,0,0



    def sa_optimaze_trapz(self, path, tac, blood, time, slices_duration,V, title,sigma,i):
         x0=[0.2,0.2,0.3,0.05,0.02]
         x0=[0.15,0.28,0.11,0.1,0.01]
##         one_div_sigma =1/sigma
##         one_div_sigma[np.isnan(one_div_sigma)]=0
##         one_div_sigma[np.isinf(one_div_sigma)]=0
##         dt=np.diff(np.insert(time,0,0))
##
##         params= blood,tac,one_div_sigma,time,dt
         
         lw = [0,0,0,0,0]

         up = [1,1,1,1,1] 
         
         ret = optimize.minimize(self.convolve_2cm_for_minimaze_trapz,x0,args=params, method='Powell', bounds=list(zip(lw, up)),
                                 options={ 'xtol': 1e-4, 'ftol':  1e-4,  'disp': False, 'return_all': False})
         #if ret.x[0]>0.99 or ret.x[1]>0.99 or ret.x[2]>0.3 or ret.x[3]>0.1 :
          #            x0=[0.2,0.2,0.3,0.05,0.02]
           #           ret = optimize.minimize(self.convolve_2cm_for_minimaze_trapz,x0,args=params, method='Powell', bounds=list(zip(lw, up)),
              #                   options={ 'xtol': 1e-4, 'ftol':  1e-1,  'disp': False, 'return_all': False})
#
 
        # ret =optimize.dual_annealing(self.convolve_2cm_for_minimaze_trapz, bounds=list(zip(lw, up)),args=params,initial_temp=50000,
         #                             no_local_search=True,maxiter=10000,restart_temp_ratio=0.000002)
         sol_tac=self.convolve_2cm([time, blood, slices_duration], ret.x[0], ret.x[1], ret.x[2], ret.x[3],ret.x[4])
         if i%100000==0:
             self.save_2plot(path, time, tac,
                       time, sol_tac,
                        title, 'time', 'con')
         if ret.x[0]>10 or ret.x[1]>10 or ret.x[2]>10 or ret.x[3]>10 :
            ret.x[0], ret.x[1], ret.x[2], ret.x[3],ret.x[4]=[0,0,0,0,0]
         return  ret.x[0], ret.x[1], ret.x[2], ret.x[3],ret.x[4]
        
         return 0,0,0,0,0

    def convolve_1cm_for_minimaze(self,X,*params):
        K1, K2,V = X        

        blood, time,tac,W,t_minus_time,dt=params
        W=W
        cp=blood
        H=K1*np.exp(-K2*(t_minus_time))
        H=H
        blood=blood  
        ct=np.sum(H*cp*dt,axis=1)
        return np.sum(W*np.power(((1-0)*ct+0*blood.T)-tac,2)) 
    def sa_optimaze_1tcm(self, path, tac, blood, time, slices_duration,V, title,W,j):
         x0=[0.5,0.07,0.01]

         dt=np.diff(np.insert(time, 0, 0))
         
 

         dt=self.convert_vector_to_matrix(dt)
         blood_opt=self.convert_vector_to_matrix(blood)
         t_minus_time=np.zeros(blood_opt.shape)
         for i,t in enumerate(time):
            t_minus_time[i,0:i+1]=t-time[0:i+1]
         time_metrix=self.convert_vector_to_matrix(time)

##         params=time,blood,tac,W


         params= blood_opt, np.asarray(time_metrix),tac,W,t_minus_time,dt
         lw = [0,0,0]
         up = [1,1,1] 
         ret = optimize.minimize(self.convolve_1cm_for_minimaze,x0,args=params, method='Powell', bounds=list(zip(lw, up)))
##                                  options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
##                                          'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
##         ret = optimize.minimize(self.convolve_1cm_for_minimaze,x0,args=params, bounds=list(zip(lw, up)))#,method="Powell")
##         ret =optimize.dual_annealing(self.convolve_1cm_for_minimaze, bounds=list(zip(lw, up)),initial_temp=1,restart_temp_ratio=0.1,maxfun=1e10,maxiter=10,accept=-100, args=params,no_local_search=False)
##         ret =optimize.dual_annealing(self.convolve_1cm_for_minimaze , bounds=list(zip(lw, up)), args=params  ,no_local_search=True)#,initial_temp=1,restart_temp_ratio=0.0008)#,no_local_search=True,initial_temp=1,restart_temp_ratio=0.1)
##        ,no_local_search=True
         if j%100000==0:
             sol_tac=self.convolve_1cm([time, blood, slices_duration], ret.x[0], ret.x[1], ret.x[2])
             self.save_2plot(path, time, tac,
                       time, sol_tac,
                        title, 'time', 'con')
         if ret.x[0]>10 or ret.x[1]>10 or ret.x[2]>10 :
            ret.x[0], ret.x[1], ret.x[2]=[0,0,0]
         return  ret.x[0], ret.x[1], ret.x[2]
        
         return 0,0,0
    
    def sa_optimaze_1tcm_trapz(self, path, tac, blood, time, slices_duration,V, title,W,i):
        x0=[0.1,0.1,0.01]
        one_div_sigma =W
        
        dt=np.diff(np.insert(time,0,0))

        params= blood,tac,one_div_sigma,time,dt
        lw = [0,0,0]

        up = [1,1,1] 
##        ret = optimize.minimize(self.convolve_1cm_for_minimaze_trapz,x0, method='Powell',args=params,bounds=list(zip(lw, up)),
##                                 options={ 'xtol': 1e-4, 'ftol': 1e-9,  'disp': False, 'return_all': False,'maxiter':100,'maxfev':1000})
##        ret =optimize.dual_annealing(self.convolve_1cm_for_minimaze_trapz, bounds=list(zip(lw, up)),args=params,initial_temp=50000,
##                                      no_local_search=True,maxiter=1000,restart_temp_ratio=0.01)
        if i%100000==0:
                self.save_2plot(path, time, tac,
                time, self.convolve_1cm([time, blood, slices_duration], ret.x[0], ret.x[1],ret.x[2]),
                title, 'time', 'con')
        if ret.x[0]>1 or ret.x[1]>1:
            ret.x[0],ret.x[1],ret.x[2]=[0,0,0]

        return ret.x[0],ret.x[1],ret.x[2]
    def two_cm_fit_p(self, path, tac, c_p, time, slices_duration,V, title,sigma,i):
        self.V=V
        #a=self.sa_optimaze('',tac, c_p,time,slices_duration,0.0149,'',1/sigma,1)
        #popt, pcov = optimize.curve_fit(self.convolve_2cm ,[time, c_p, slices_duration], tac,bounds=[0,1])
        ind=sum(sigma<10)
        popt=[0,0,0,0,0]
        try:

            popt, pcov = optimize.curve_fit(self.convolve_2cm ,[time, c_p, slices_duration], tac)
        except:
            popt=[0,0,0,0,0]
        if i%100000==0:    
             self.save_2plot(path, time, tac,
                           time, self.convolve_2cm([time, c_p, slices_duration], popt[0], popt[1], popt[2], popt[3],popt[4]),
                            title, 'time', 'con')

        return popt[0], popt[1], popt[2], popt[3], popt[4]
    def get_integral_to_all_t(self,y,x):
        results=[]
        for i in range(len(x)):
            results.append(np.trapz(y[:i+1],x[:i+1]))
        return np.asarray(results)
            
    

    def fit_two_tcm_nnls(self, path, tac, c_p, time, slices_duration,V, title,sigma,i):
        one_integral_cp=self.get_integral_to_all_t(c_p,time)
        twice_integral_cp=self.get_integral_to_all_t(one_integral_cp,time)
        one_integral_tac=self.get_integral_to_all_t(tac,time)
        twice_integral_tac=self.get_integral_to_all_t(one_integral_tac,time)
        X=np.asarray([c_p,one_integral_cp,twice_integral_cp,one_integral_tac,twice_integral_tac]).T
        P1,P2,P3,P4,P5=optimize.nnls(X,tac)[0]
        K1=(P1*P4+P2)/(1-P1)
        K2=-((P1*P5+P3)/(P1*P4+P2))-P4
        K4=-(P5/K2)
        K3=-(K2+K4+P4)
        return K1,K2,K3,K4,P1

    def two_cm_fit(self, path, tac, c_p, time, slices_duration,V, title,sigma,i):
        self.V=V
        #a=self.sa_optimaze('',tac, c_p,time,slices_duration,0.0149,'',1/sigma,1)
        #popt, pcov = optimize.curve_fit(self.convolve_2cm ,[time, c_p, slices_duration], tac,bounds=[0,1])
        ind=sum(sigma<10)

       #ftol=2e-1, xtol=1e-2
        
        popt=[0,0,0,0,0]
        try:

            popt, pcov = optimize.curve_fit(self.convolve_2cm, [time, c_p, slices_duration], tac,
                                            p0=[0.7,0.2,0.5,0.05,0.01],maxfev=100000,ftol=1e-8, xtol=1e-8)
        except:
            popt=[0,0,0,0,0]
        
        if popt[0]<0 or popt[1]<0 or popt[2]<0 or popt[3]<0 :
            popt=[0,0,0,0,0]
        '''
        K1 = (popt[2] + popt[3]) / (1 - 0.04)

        K2 = (popt[2] * popt[0] + popt[3] * -popt[1]) / (popt[2] + popt[3])

        K4 = (popt[0] * popt[1]) / K2

        K3 = (popt[0] + popt[1]) - K2 - K4
        '''
        if i%100000==0:    
             self.save_2plot(path, time, tac,
                           time, self.convolve_2cm([time, c_p, slices_duration], popt[0], popt[1], popt[2], popt[3],popt[4]),
                            title, 'time', 'con')

        return popt[0], popt[1], popt[2], popt[3], popt[4]

    def get_x_non_linear_patlak(self, data, kb):

        time = data[0]

        blood = data[1]

        slicesDuration = data[2]

        results = np.asarray([])

        for i, t in enumerate(time):
            i = i + 1

            tua = time[0:i]

            tua = tua

            dtua = np.diff(np.insert(tua, 0, 0))

            c_p = blood[0:i]

            f_t1 = np.exp(-kb * (t - tua)) * c_p

            results = np.append(results, np.sum(f_t1 * dtua))

        return results / blood

    def fit_patlak(self, x, y):

        p = np.polyfit(x, y, 1)

        z = np.polyval(p, x)

        r = 1 - (np.sum(np.power(y - z, 2) / np.sum(np.power(y - np.mean(z), 2))))

        return p, r

    def non_linear_patlak(self,ii, data):

        time = data[0]

        blood = data[1]


        slicesDuration = data[2]

        TAC = data[3]

        numerator = np.asarray([])

        kb_guass = np.linspace(0,0.1,1000)

        r = 0

        k_b = -1

        p = [0, 0]

        x_linear = -1

        t_start = 0

        for k_b_temp in kb_guass[1:]:

            x = self.get_x_non_linear_patlak(data, k_b_temp)

            for num in range(np.argmax(c_p)+1,21): #21

                try:
                    p_temp, r_temp = self.fit_patlak(x[num:], TAC[num:] / blood[num:])
                except:                    p_temp, r_temp=[0,0],0
                

                if r_temp > r and p_temp[0]/k_b_temp>10:
                    #print(r_temp)

                    r = r_temp

                    k_b = k_b_temp

                    p = p_temp

                    x_linear = x

                    t_start = num
                        
                
              
                  
            
                    

        if r<0:
            return 0, 0, [0,0], x_linear, t_start
        else:
            #self.save_plot_patlak(x_linear, TAC / blood, x_linear, np.polyval(p, x_linear), p[0], k_b, r,'D:','patlak')
            return k_b, r, p, x_linear, t_start

    def non_linear_patlak2(self, data, k_b):

        time = data[0]

        blood = data[1]

        slicesDuration = data[2]

        TAC = data[3]

        numerator = np.asarray([])

        r = 0.5

        p = [0, 0]

        x_linear = -1

        t_start = 0

        x = self.get_x_non_linear_patlak(data, k_b)

        for num in range(np.where(blood < 10)[0][-1] + 1, 20):

            p_temp, r_temp = self.fit_patlak(x[num:], TAC[num:] / blood[num:])

            if r_temp > r:
                print(r_temp)

                r = r_temp

                p = p_temp

                x_linear = x

                t_start = num

        return k_b, r, p[0], x_linear, t_start

    def save_plot_patlak(self, x1, y1, x2, y2, ki, k_loss, r, path, title):

        fig1, ax1 = plt.subplots()

        ax1.plot(x1, y1, 'ro', x2, y2)

        ax1.set_title(" ki={:.3f} ".format(ki) + "k_loss= {:.3f} ".format(k_loss) + "r= {:.3f} ".format(r))

        ax1.set_xlabel('time')

        ax1.set_ylabel('concentration')

        fig1.savefig(path + '/' + title + '.png')
        plt.close('all')


    def __logan_fit(self,x, y):
        p = np.polyfit(x, y, 1)
        z = np.polyval(p, x)
        r = 1 - (np.sum(np.power(y - z,2) / np.sum(np.power(y - np.mean(z),2 ))))
        return p, r

    def __integral(self,x, y):
        return np.cumsum(x * y)

    def logan_for_one_tac(self,ii,ct, cp, min_r_squared, slice_times):
        #min_index=np.where(cp==np.max(cp))[0][0]+1
        if True:#min_index<14:
            r=0
            cp_integral = self.__integral(slice_times, cp);
            ct_integral = self.__integral(slice_times, ct);
            full_x = cp_integral / ct;
            full_y = ct_integral / ct;
            p_sol = [0, 0]
            final_r_squared=0
            
            for i in range(8,22):#21
                try:
                    p, r = self.__logan_fit(full_x[i:len(full_x)], full_y[i:len(full_y)])
                except:
                    r=0

                if r > min_r_squared and p[0]>0  :
                    min_r_squared = r
                    final_r_squared = r
                    p_sol = p
                    index=i
                    if r>0.7:
                        break
                
                   
        if final_r_squared:
            return p_sol,final_r_squared,cp_integral,ct_integral,index
        return [0,0],0,0,0,0


    def save_plot_logan(self,x1,y1,x2,y2,k,r,path,title):
            fig1, ax1 = plt.subplots()
            ax1.plot(x1,y1,'ro',x2,y2)
            ax1.set_title(" k_logan={:.3f} ".format(k)+"r= {:.3f} ".format(r))
            fig1.savefig(path+'/'+title)
            plt.close('all')


    def __logan_fit(self,x, y):
        p = np.polyfit(x, y, 1)
        z = np.polyval(p, x)
        r = 1 - (np.sum(np.power(y - z,2) / np.sum(np.power(y - np.mean(z),2 ))))
        return p, r

    def __integral(self,x, y):
        return np.cumsum(x * y)

    def patlak_for_one_tac(self,ii,ct, cp, min_r_squared, slice_times):
        #min_index=np.where(cp==np.max(cp))[0][0]+1
        if True:#min_index<14:
         
            r=0
            cp_integral = self.__integral(slice_times, cp);
            full_y= ct / cp;
            full_x= cp_integral / cp;
            p_sol = [0, 0]
            final_r_squared=0
            
            for i in range(10,22):#21
                try:
                    p, r = self.__logan_fit(full_x[i:len(full_x)], full_y[i:len(full_y)])
                except:
                    r=0

                if r > min_r_squared and p[0]>0  :
                    min_r_squared = r
                    final_r_squared = r
                    p_sol = p
                    index=i
                    if r>0.9:
                        break
                
                   
        if final_r_squared>0.:
            return p_sol,final_r_squared,full_x,full_y
        return [0,0],0,0,0


    def save_plot_patlak(self,x1,y1,x2,y2,k,r,path,title):
            fig1, ax1 = plt.subplots()
            ax1.plot(x1,y1,'ro',x2,y2)
            ax1.set_title(" patlak={:.3f} ".format(k)+"r= {:.3f} ".format(r))
            fig1.savefig(path+'/'+title)
            plt.close('all')

