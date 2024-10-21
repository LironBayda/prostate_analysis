 # This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import pandas as pd
from dynamic_pet_models.general_functions import *
import numpy as np
from torch import nn
import torch 
import torch.optim as optim
from dynamic_pet_models.bilateral_filter import BilateralFilter
from scipy import stats

import matplotlib.pyplot as plt

from scipy import optimize
ephocs=5000


def wavelet_denoising(x, wavelet='db4'):
     
    coeff = pywt.wavedec(x, wavelet, mode="constant")
    sigma=stats.median_abs_deviation(np.asarray(coeff).flatten())/0.6745
    N =len(x.flatten())
    M=len(coeff[1:])
    #uthresh = sigma/stats.norm.cdf((1-0.05)/(2*M))
    uthresh= sigma*np.sqrt(2*np.log(N))
    #coeff[0] = pywt.threshold(coeff[0], value=uthresh, mode='hard')
    coeff[1:] = (pywt.threshold(i, value=   uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='constant')

def plot_all( path, time, IDIF,cancer_p,cancer_f,notcancer_p,not_cancer_f, title, x_label, y_label):

        print('plot')

        fig, ax = plt.subplots()
        ax.plot( time,np.asarray(cancer_p)/1000, 'ro')

        ax.plot( time,np.asarray(notcancer_p)/1000, 'g^')
        ax.plot(time, IDIF/1000, 'b')
        ax.plot(time, np.asarray(cancer_f)/1000,'pink')
        ax.plot(time, np.asarray(not_cancer_f)/1000,'y')

        ax.set_title(title)

        ax.set_xlabel(x_label)

        ax.set_ylabel(y_label)
        ax.legend([ 'suspect lesion', 'reference','IDIF'])
        fig.savefig(join(path, title + ".png"))
        plt.show()


def save_2plot(path, x1, y1, x2, y2, title, x_label, y_label):

        print('plot')

        fig, ax = plt.subplots()

        ax.plot(x1, y1, 'ro', x2, y2)

        ax.set_title(title)

        ax.set_xlabel(x_label)

        ax.set_ylabel(y_label)

        fig.savefig(join(path, title + ".png"))
        plt.close('all')

def convolve_1cm( data, K1, K2):
        time = np.asarray(data[0])

        blood = np.asarray(data[1])
        cp=blood
        theta=K2
        phi=K1
        ct=[]
        for i,t in enumerate(time):
            H=phi*np.exp(-theta*(t-time[0:i+1]))
            ct.append((1-0)*np.trapz(H*cp[0:i+1],x=time[0:i+1])+0*blood[i])
        return ct



def convolve_2cm(data, K1, K2, K3, K4,V):
        time = np.asarray(data[0])

        blood = np.asarray(data[1])
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
            ct.append(np.trapz(H1*cp[0:i+1],x=time[0:i+1])
                      +np.trapz(H2*cp[0:i+1],x=time[0:i+1]))
        return np.asarray(ct)





def my_loss_part1a(output1, output2,target1):
    ct=output1[:,:,0]+output1[:,:,1]
    return  0.1*torch.mean(torch.pow(ct-target1,2))+torch.mean(torch.pow(output2[-1]-target1,2))
def my_loss_part1b(output1, target1):     
    return torch.mean(torch.pow(output1[:,:,2]-target1,2))



            
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1,61)
        self.convLayer1 =nn.Conv1d(1,12,kernel_size=2)#,stride=2)
        self.maxPool1d1 =nn.MaxPool1d(2,return_indices=True)
##
        self.convLayer2 =nn.Conv1d(12, 18, kernel_size=2)#,stride=2)
        self.maxPool1d2 =nn.MaxPool1d(2,return_indices=True)
        self.convLayer3 =nn.Conv1d(18,18, kernel_size=2)#,stride=2)
        self.maxPool1d3 =nn.MaxPool1d(2,return_indices=True)

        self.flatten =nn.Flatten(start_dim=1, end_dim=- 1)
        self.fcd=nn.Linear(108,60, bias=True)
        self.fcu=nn.Linear(60,108, bias=True)
        self.unflatten3 =nn.Unflatten(1, (18,-1))
        self.maxUnPool1d3 =nn.MaxUnpool1d(2)
 
        self.convTransposeLayer3 =nn.ConvTranspose1d(18,18, kernel_size=3)#,stride=2)
        self.dropout =nn.Dropout(0.5)
        self.maxUnPool1d2 =nn.MaxUnpool1d(2)
        self.convTransposeLayer2 =nn.ConvTranspose1d(18,12,kernel_size=3)#,stride=2)
        self.dropout1 =nn.Dropout(0.6)
        self.maxUnPool1d1 =nn.MaxUnpool1d(2)
        self.convTransposeLayer1 =nn.ConvTranspose1d(12 ,1,kernel_size=2)#,stride=2)
        self.output = nn.Linear(61,3)
    def get_code(self,e):
        return self.code

    def forward(self, features):


        features = self.input(features)
        activation = self.convLayer1(features)
        activation = torch.tanh(activation)

        activation, indices1 = self.maxPool1d1(activation)
        activation = self.dropout(activation)



        
        activation = self.convLayer2(activation)
        activation = torch.tanh(activation)

        activation, indices2 = self.maxPool1d2(activation)
##
        activation = self.dropout(activation)




        activation = self.convLayer3(activation)
        activation = torch.tanh(activation)
####
        activation, indices3 = self.maxPool1d3(activation)
        activation = self.dropout(activation)


        activation = self.flatten(activation)
        activation  = self.fcd(activation)
        self.code = torch.tanh(activation)
        activation = self.fcu(self.code)
        activation = torch.tanh(activation)
        activation = self.unflatten3(activation)
        
        activation = self.maxUnPool1d3(activation, indices3)
        activation = self.dropout(activation)
       
        activation = self.convTransposeLayer3(activation)
        activation = torch.tanh(activation)

        activation = self.maxUnPool1d2(activation,indices2)
        activation = self.dropout(activation)

        activation = self.convTransposeLayer2(activation)
        activation = torch.tanh(activation)


        activation = self.maxUnPool1d1(activation, indices1)
        activation = self.dropout(activation)

        activation = self.convTransposeLayer1(activation)

        reconstructed = torch.tanh(activation)
        reconstructed = self.output(reconstructed)
        reconstructed = torch.tanh(reconstructed)
        return reconstructed
##

      
class Net2(nn.Module):
        def __init__(self,blood):
                super().__init__()
                self.Ks =(torch.tensor(0.136,requires_grad=True),torch.tensor(0.277,requires_grad=True)
                          ,torch.tensor(0.108,requires_grad=True))
                self.time=np.asarray([0,10,20,30,40,50,60,90,120,150,180,210,240,270,300,330,360,410,460,510,560,610,660,960,1200])/60
                self.slices_duration=np.asarray([10,10,10,10,10,10,30,30,30,30,30,30,30,30,30,30,30,50,50,50,50,50,300,240])/60
                self.time_matrix=self.convert_vector_to_matrix(self.time)

                self.dt=torch.tensor(self.convert_vector_to_matrix(self.slices_duration))
                self.blood=torch.tensor(self.convert_vector_to_matrix(np.insert(blood,0,0)))
                self.t_minus_time=np.zeros(self.time_matrix.shape)
                for i,t in enumerate(self.time):
                        self.t_minus_time[i,0:i+1]=t-self.time[0:i+1]
                self.t_minus_time=torch.tensor(self.t_minus_time)
                #time_opt=self.convert_vector_to_matrix(time)
        def convolve_2cm_for_minimaze(self):
                delta=torch.sqrt(torch.pow(self.Ks[1]+self.Ks[2],2))

                theta1=(self.Ks[1]+self.Ks[2]+delta)/2
                theta2=(self.Ks[1]+self.Ks[2]-delta)/2
                phi1=(self.Ks[0]*(theta1-self.Ks[2]))/delta
                phi2=(self.Ks[0]*(theta2-self.Ks[2]))/(-delta)
                H1=phi1*torch.exp(-theta1*(self.t_minus_time))
                H2=phi2*torch.exp(-theta2*(self.t_minus_time))

                ct=torch.sum(H1[1:,1:]*self.blood[1:,1:]*self.dt,axis=1)\
                    +torch.sum(H2[1:,1:]*self.blood[1:,1:]*self.dt,axis=1)
                return ct          
                               

        def parameters(self):
                return iter(self.Ks)
        def convert_vector_to_matrix(self,vector):
                matrix=np.zeros((vector.shape[0],vector.shape[0]))
                for i,x in enumerate(vector):
                    matrix[i:,i]=x
                return matrix

        def forward(self,C0,C1,C2,dC1,dC2):
                ct=[]
                self.ct=self.convolve_2cm_for_minimaze()
                
                x=[(dC1+dC2)-(self.Ks[0]*C0-(self.Ks[1])*C1),
                    (dC2)-self.Ks[2]*C1,

                   (dC1)-(self.Ks[0]*C0-(self.Ks[1]+self.Ks[2])*C1),self.ct/num]
#-self.Ks[2]*C1,
                    
                return x

def div(x,y):
    div=torch.autograd.grad(
                y, x,
                grad_outputs=torch.ones_like(y),
                retain_graph=True,
                create_graph=True

                )[0]

    return div

def my_loss_part2(output):
    #loss= torch.sum(torch.pow(output,2))
    #loss=torch.reshape(loss, (1, 1))
    return  torch.mean(torch.pow(output[1],2))+torch.mean(torch.pow(output[0],2))#+\
       # torch.mean(torch.pow(output[0],2))
   


def my_loss_part_initi_con(C1_0):
    #loss= torch.sum(torch.pow(output,2))
    #loss=torch.reshape(loss, (1, 1))

    return torch.pow(C1_0,2)


def my_loss_part_reg(C1):
    return torch.pow(torch.min(torch.cat([ torch.tensor([0]), C1], dim=0),0)[0],2)
class PhysicsInformedNN():
        def __init__(self, t, tac, c_p, num):
                
        
                # data
                self.t =t
                self.dC1 = 0
                self.tac_norm = torch.tensor(tac,dtype=torch.float,requires_grad=False)/num
                self.c_p_norm = torch.tensor(c_p,dtype=torch.float,requires_grad=False)/num
                self.num = num
                self.iter=0
                # deep neural networks
                self.net1 =Net1()
                self.net2 =Net2(c_p)
        def get_code(self):
                X1=torch.reshape(self.t,(24,1,1))
                return self.net1.get_code(X1)
        def suffle(self,x,y,z):
                index=np.random.shuffle(list(range(24)))
                return x[index],y[index],z[index]
        def predict(self):
                X1=torch.reshape(self.t,(24,1,1))
                return  self.net1(X1)
        def loss(self):
                X1=torch.reshape(self.t,(24,1,1))
             
                output1 = self.net1(X1)
                self.dC1 = 1/(sigma_mid)*torch.autograd.grad(
                output1[:,:,0], self.t,
                grad_outputs=torch.ones_like(output1[:,:,0]),
                retain_graph=True,
                create_graph=True
                )[0]
                self.dC2 = 1/(sigma_mid)*torch.autograd.grad(
                output1[:,:,1], self.t,
                grad_outputs=torch.ones_like(output1[:,:,1]),
                retain_graph=True,
                create_graph=True
                )[0]
                self.dC1=torch.reshape(self.dC1,(24,1,1))
                self.dC2=torch.reshape(self.dC2,(24,1,1))

                num_reg=10
                num_init=10

                output2 =  self.net2(self.c_p_norm,output1[:,:,0],output1[:,:,1],self.dC1,self.dC2)
                loss1=my_loss_part1a(output1,output2, self.tac_norm)\
                +my_loss_part1b(output1,self.c_p_norm)\
                +num_init*my_loss_part_initi_con(output1[:,:,0][0])+num_init*my_loss_part_initi_con(output1[:,:,1][0])\
                +num_reg*my_loss_part_reg(output1[:,0,0])+num_reg*my_loss_part_reg(output1[:,0,1])
                loss2=my_loss_part2(output2)
                loss=loss1+loss2
                loss.backward()
                self.iter += 1
                if self.iter % 1000== 0:
                    for k in self.net2.parameters():
                                    print(k)
                    print(
                        'Iter %d, loss1: %.5e loss2: %.5e' % (self.iter, loss1.item(), loss2.item())
                    )
                return loss
            
        def loss2(self):
                X1=torch.reshape(self.t,(1,24))
             
                output1 = self.net1(X1)
                self.dC1 = 1/(sigma_mid)*torch.autograd.grad(
                output1[:,:,0], self.t,
                grad_outputs=torch.ones_like(output1[:,:,0]),
                retain_graph=True,
                create_graph=True
                )[0]
                self.dC2 = 1/(sigma_mid)*torch.autograd.grad(
                output1[:,:,1], self.t,
                grad_outputs=torch.ones_like(output1[:,:,1]),
                retain_graph=True,
                create_graph=True
                )[0]
                output2 =  self.net2(self.c_p_norm,output1[:,:,0],output1[:,:,1],self.dC1,self.dC2)
            
                loss2=my_loss_part2(output2)
                
                return loss2

        def set_optim(self,lr):
                self.optimizer = torch.optim.Adam(
                    list(self.net1.parameters())+list(self.net2.parameters()),
                    lr=lr
                )
           

        def train(self):
                #self.net1.train()
               # self.net2.train()
                self.optimizer.zero_grad()
                # Backward and optimize
                loss=self.optimizer.step(self.loss)
                #loss=self.optimizer2.step(self.loss)

                return loss



class EarlyStopper:
    def __init__(self, patience = 50 , min_delta  = 0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
           # self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta*validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
def scheduler(epoch, lr):   
        if epoch< 0:# or (epoch >100 and epoch <2000) or epoch >2060 :
            return 0.01#lr
      
        elif epoch<0:
            return 0.01
        elif epoch<0:
            return 0.0001
        else:
            return 0.001
###############################6##########################################################################################
if __name__ == '__main__':


    mypath= 'C:\\Users\\liron\\Downloads\\copy\\'
    paths = [join(mypath, f) for f in listdir(mypath) if f.startswith('sub')]
    data = {}
    data2 = {}
    data3 = {}
    data4 = {}
    data5 = {}
    data6 = {}
    count=0
    
    for path in paths:
        
        __big_slice, __end_slice = 0, 24
        dynamic_pet_path = os.path.join(path, 'dynPET')
        dce_path = os.path.join(path, 'DCE')

##        if not  os.path.exists(os.path.join(dce_path, 'rpcancer_dixon.nii')):
##                continue
        #data_from_dicom = dynamic_pet.get_variables_from_dicom(dynamic_pet_path, __big_slice, __end_slice)
        slices_duration=np.asarray([10,10,10,10,10,10,30,30,30,30,30,30,30,30,30,30,30,50,50,50,50,50,300,240])/60
        mid=np.asarray([10,20,30,40,50,60,90,120,150,180,210,240,270,300,330,360,410,460,510,560,610,660,960,1200])/60
        images = []
   
        artery_mask = get_mask(dynamic_pet_path, 'pvc_blood_mask.nii')
        c_p = get_c_p(mid, artery_mask, __big_slice, __end_slice, dynamic_pet_path, 'motcorrW.nii.g')*1.62
 
        images = get_images(dynamic_pet_path, 'motcorrW.nii.g', __big_slice, __end_slice)
        print(path)
       

##########################################################################################################################
        cancer_mask = get_mask(dynamic_pet_path, 'pvc_cancer_dixon.nii')
        tac = get_tac(images, cancer_mask)
        num=np.max([np.max(tac),np.max(c_p)])
        early_stopper = EarlyStopper()

        sigma_mid=torch.tensor(np.std(mid),dtype=torch.float,requires_grad=False)
        t=torch.tensor((mid-np.mean(mid)/np.std(mid)),dtype=torch.float,requires_grad=True)
        pinn=PhysicsInformedNN(t, tac, c_p, num)
        lr=0.000001
        loss=90000
        epochs=10000
        K_cancer=[]

        for epoch in range(epochs):
            K_cancer=[]
            lr=scheduler(epoch, lr)
            pinn.set_optim(lr)
            loss=pinn.train() 
            if epoch>0 and early_stopper.early_stop(loss):             
                for k in pinn.net2.parameters():
                    K_cancer.append(k.detach().numpy())
                if K_cancer[0]>0 and K_cancer[1]>0 and K_cancer[2]>0 and K_cancer[0]<1 and K_cancer[1]<1 and K_cancer[2]<1:
                    break
##                               

        r=pinn.predict().detach().numpy().T*num
        save_2plot(dynamic_pet_path, mid, tac,mid, r[0][0]+r[1][0],'nn 2tcm_cancer', 'time', 'con') 
##        save_2plot(dynamic_pet_path, mid, c_p,mid, r[2][0],'nn 2tcm_blood_cancer', 'time', 'con') 
        save_2plot(dynamic_pet_path, mid, r[0][0],mid, r[1][0],'nn 2tcm_C1_C2_cancer', 'time', 'con') 
        K_cancer=[]


        for k in pinn.net2.parameters():
                K_cancer.append(k.detach().numpy())
        print(K_cancer)
##        code_cancer=pinn.get_code().detach().numpy()[0]
        sol_tac=convolve_2cm([mid, c_p, 0], K_cancer[0], K_cancer[1], K_cancer[2], 0,0)
        save_2plot(dynamic_pet_path, mid, tac,mid, sol_tac,'nn 2tcm_K_cancer', 'time', 'con') 


