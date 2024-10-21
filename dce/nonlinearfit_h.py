import numpy as np

import nibabel as nib

from os.path import join

from os import listdir

import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt
from scipy import optimize
from general_functions import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MASK_CANCER=['rpcancer_di','nii']
MASK_NOTCANCER=['rpnotcancer_dix','nii']

def save_2plot(path, x1, y1, x2, y2, title, x_label, y_label):

    fig, ax = plt.subplots()

    ax.plot(x1, y1, 'ro', x2, y2)

    ax.set_title(title)

    ax.set_xlabel(x_label, fontsize=40)
    ax.set_ylabel(y_label, fontsize=40)

    fig.savefig(join(path, title + ".png"))
    plt.close('all')


def save_plot(path, x1, y1, title, x_label, y_label):

    fig, ax = plt.subplots()

    ax.plot(x1, y1)

##    ax.set_title(title)

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    fig.savefig(join(path, title + ".png"))
    plt.close('all')


def get_images(mypath, prefix, big_slice, end_slice):
##    dynamicImaging = [join(mypath, f) for f in listdir(mypath) if  f.startswith(prefix[0])   and prefix[1] in f]
    dynamicImaging = [join(mypath, f) for f in listdir(mypath) if  prefix[0] in f   and prefix[1] in f]

    images = [nib.load(path).get_fdata() for path in dynamicImaging[big_slice:end_slice]]
    
    images = np.asarray(images)

    images[np.isnan(images)] = 0
    images[np.isinf(images)] = 0

    return images, nib.load(dynamicImaging[0]).affine


def save_img(img, affine, name):
    array_img = nib.Nifti1Image(img.astype(np.float64), affine)

    nib.save(array_img, name)


def get_t0(t1_0,data):
    deg1 = data[0]

    deg2 = data[1]

    sol = data[2]


    tr = 5.18 * 0.001

    numerator1 = np.sin(deg1) * (1 - np.exp(-(tr / t1_0)))

    denominator1 = 1 - np.cos(deg1) * (np.exp(-(tr / t1_0)))

    numerator2 = 1 - np.cos(deg2) * (np.exp(-(tr / t1_0)))

    denominator2 = np.sin(deg2) * (1 - np.exp(-(tr / t1_0)))

    return np.power(((numerator1 / denominator1) * (numerator2 / denominator2))-sol,2)


def get_t_obs(t1_obs,data):
    #s_dyn[i] / s15
    deg1 = data[0]

    deg2 = data[1]

    t1_0= data[2]

    sol = data[3]


    tr = 5.18 * 0.001


    
    numerator1 = np.sin(deg1) * (1 - np.exp(-(tr / t1_obs)))

    denominator1 = 1 - np.cos(deg1) * (np.exp(-(tr / t1_obs)))

    numerator2 = 1 - np.cos(deg2) * (np.exp(-(tr / t1_0)))

    denominator2 = np.sin(deg2) * (1 - np.exp(-(tr / t1_0)))

    return np.power(((numerator1 / denominator1) * (numerator2 / denominator2))-sol,2)





##

def get_TAC_from_mask(images, mask_pre):
  
    mask, affine = get_images(path, mask_pre, 0, 1)

    images_temp=[(images[:,:,:,:,s]-images[:,:,:,:,0])/images[:,:,:,:,0] for s in range(images.shape[4])]
    ####  
    images_temp = np.asarray(np.transpose(images_temp,(1,2,3,4,0)))
    images_temp[np.isnan(images_temp)] = 0
    images_temp[np.isinf(images_temp)] = 0
    ##        
    ##########    
    TAC = np.asarray([np.mean(images_temp[:, :, :, :, i][mask>0]) for i in range(35)])
    TAC[TAC<0]=0
##    TAC = np.asarray([np.mean(np.sort(images_temp[:, :, :, :, i][mask>0])[-int(np.sum(mask)*0.25):]) for i in range(35)])


##           
    return TAC,0


def get_sigma_from_mask(images, mask_pre):
    mask, affine = get_images(path, mask_pre, 0, 1)

    sigma = np.asarray([np.median(images[:, :, :, :, i][mask>0]) for i in range(35)])

   
    return sigma

def get_region_from_mask(image, mask_pre):

    mask, affine = get_images(path, mask_pre, 0, 1)

    region = np.mean(image[mask>0])
    sigma = np.std(image[mask>0])


    return region




from sklearn.preprocessing import normalize
def get_wash_in_washout_for_TAC(TAC):

   

    TAC[TAC < 0] = 0
    time = get_time()*60

    big = 5  # np.where(TAC>=0.01)[0][0]

    p1 = np.polyfit(time[big:big + 4], TAC[big:big + 4], deg=1)

    p2 = np.polyfit(time[20:], TAC[20:], deg=1)

    return p1, p2,TAC

def get_time():
        return np.cumsum([7.55/60] * 35)





def get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image, mask_per,):

    s_dyn,j = get_TAC_from_mask(dyn_image, mask_per)

    s_dyn = np.asarray(s_dyn)


 
    return s_dyn,j



def get_tac_all_from_mask_blood(path, img2_path_per, img15_path_per, dyn_image, mask_per,):
    img2, affine = get_images(path, img2_path_per, 0, 1)

    img15, affine = get_images(path, img15_path_per, 0, 1)

    s2 = get_region_from_mask(img2, mask_per)

    s15 = get_region_from_mask(img15, mask_per)

    t1_0 = optimize.fsolve(get_t0,[1],args=([ 2 * 0.017,15 * 0.017,s2 /s15 ]),factor=0.5)



    t1_obs = np.asarray([])

    s_dyn,j = get_TAC_from_mask(dyn_image, mask_per)
    s_dyn = np.asarray(s_dyn)
##    iii=np.argmax(s_dyn)

    for i in range(dyn_image.shape[4]):
        popt = optimize.fsolve(get_t_obs,[1], args=([15 * 0.017,2 * 0.017 , t1_0,s_dyn[i] / s2]),factor=1)

        t1_obs = np.append(t1_obs, popt)
    

    TAC = 1 / (4.5) * (1 / t1_obs - 1 / t1_0)

##    TAC[:iii-2]=0
    TAC[TAC<0]=0
    TAC[TAC>800]=0

    return TAC,j





##def fit_model_h(time, K_in, K_out,k12,k21):
##    
##
##
## return 1+a


from sklearn.metrics import r2_score



                                      
def all_from_mask_model(data,data2, path, img2_path_per, img15_path_per, dyn_image,ii,aif):

    TAC_cancer,j_cancer = get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image, MASK_CANCER)

    TAC_notcancer,j_notcancer = get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image, MASK_NOTCANCER)

    time = get_time()  
    K_tras_cancer, Ve_cancer = get_ks(time, aif, TAC_cancer,ii)
    K_kep_cancer=K_tras_cancer/Ve_cancer


    K_tras_notcancer, Ve_notcancer = get_ks(time, aif, TAC_notcancer,ii)
    K_kep_notcancer=K_tras_notcancer/Ve_notcancer




    
    save_plot(path, time, aif ,'aif', '', '')
  

   
    save_2plot(path, time, TAC_cancer, time, fit_model([time, aif], K_tras_cancer, K_kep_cancer)
               
               ,'K_tras_cancer ' + str(K_tras_cancer) + 'K_kep_cancer ' + str(K_kep_cancer), '', '')
    save_2plot(path, time, TAC_notcancer, time, fit_model([time, aif], K_tras_notcancer, K_kep_notcancer), 
            'K_tras_notcancer ' + str(K_tras_notcancer) + 'K_kep_notcancer ' + str(K_kep_notcancer), '', '')

    data[path.split('\\')[-2]] = {'K_tras_cancer': K_tras_cancer, 'Ve_cancer': Ve_cancer,'K_kep_cancer':K_kep_cancer,
                              'K_tras_notcancer': K_tras_notcancer, 'Ve_notcancer':Ve_notcancer,'K_kep_notcancer': K_kep_notcancer,
                                 'K_tras_r': K_tras_cancer/K_tras_notcancer, 'Ve_r': Ve_cancer/Ve_notcancer,'K_kep_r':K_kep_cancer/K_kep_notcancer
                                  }

    
    return data,[]


def all_from_mask_model_img(data, path, img2_path_per, img15_path_per, dyn_img,ii):
##
##    img2, affine = get_images(path, img2_path_per, 0, 1)
##    img15, affine = get_images(path, img15_path_per, 0, 1)
    K_tras=np.zeros((dyn_img.shape[1],dyn_img.shape[2],dyn_img.shape[3]))
    Ve=np.zeros((dyn_img.shape[1],dyn_img.shape[2],dyn_img.shape[3]))
    #wash_in,affine= get_images(path, 'wash_in.nii', 0, 1)
    mask, affine = get_images(path, MASK_CANCER, 0, 1)
    mask2, affine = get_images(path, MASK_NOTCANCER, 0, 1)
    mask=mask+mask2
    #th=np.sort(wash_in.flatten())
    #th=th[-int(len(th)/100)*10]
    #print(th)

    time = get_time()
    _,x_size,y_size,z_size,_=dyn_img.shape
    nonI=0
    for i in range(x_size):
        for j in range(y_size):    
            for k in range(z_size):
                if mask[0,i,j,k]>0:#  and img2[0,i,j,k]>0 and img15[0,i,j,k]>0:
                    TAC= dyn_img[0,i,j,k,:]#get_tac_all_from_mask_img(path, img2[0,i,j,k], img15[0,i,j,k], dyn_img[0,i,j,k,:])
                    K_tras[i,j,k], Ve[i,j,k] = get_ks(time, aif, TAC,ii)
                    if nonI==0:
                        save_2plot(path, time, TAC, time, fit_model([time, aif], K_tras[i,j,k], K_tras[i,j,k]/Ve[i,j,k]), 
                         'Ve[i,j,k] ' + str(Ve[i,j,k]) + 'K_tras[i,j,k] ' + str(K_tras[i,j,k]), '', '')
                        nonI=2

                


        save_img(K_tras, affine, join(path,'K_tras.nii'))
        save_img(Ve, affine, join(path,'Ve.nii'))
    save_plot(path, time, aif ,'aif', '', '')


    return data




def all_from_mask_contrast(data,data2, path, img2_path_per, img15_path_per, dyn_image):

    dyn_image, affine = get_images(path, dyn_image_per, 0, 1)

    TAC_cancer,_ =get_TAC_from_mask(dyn_image,
                                           MASK_CANCER)  # get_tac_all_from_mask(path,img2_path_per,img15_path_per,dyn_image_per,'rcancer_native')

    TAC_cancer= np.asarray(TAC_cancer)

    TAC_notcancer,_ = get_TAC_from_mask(dyn_image,
                                                  MASK_NOTCANCER)  # get_tac_all_from_mask(path,img2_path_per,img15_path_per,dyn_image_per,'rnotcancer_native')

    TAC_notcancer= np.asarray(TAC_notcancer)


    time = get_time()*60

    wash_in_cancer, wash_out_cancer,TAC_cancer = get_wash_in_washout_for_TAC(TAC_cancer)

    wash_in_notcancer, wash_out_notcancer,TAC_notcancer = get_wash_in_washout_for_TAC(TAC_notcancer)


    save_2plot(path, time, TAC_notcancer, time[1:10], np.polyval(wash_in_notcancer, time)[1:10],
            'wash_in_notcancer ' + str(wash_in_notcancer), '', '')

    save_2plot(path, time, TAC_notcancer, time, np.polyval(wash_out_notcancer, time),
            'wash_out_notcancer' + str(wash_out_notcancer), '', '')

    save_2plot(path, time, TAC_cancer, time[1:10], np.polyval(wash_in_cancer, time)[1:10],
            'wash_in_cancer ' + str(wash_in_cancer), '', '')

    save_2plot(path, time, TAC_cancer, time, np.polyval(wash_out_cancer, time, ),
            'wash_out_cancer ' + str(wash_out_cancer), '', '')

    data[path.split('\\')[-2]] = {'wash_in_cancer': wash_in_cancer[0], 'wash_out_cancer': wash_out_cancer[0],
                              'wash_in_notcancer': wash_in_notcancer[0], 'wash_out_notcancer': wash_out_notcancer[0]}

    data2['sample'+str(__COUNT1)] = {'wash_in': wash_in_cancer[0], 'wash_out': wash_out_cancer[0],'cancer':1}
                              

    data2['sample'+str(__COUNT1+1)] = {'wash_in': wash_in_notcancer[0], 'wash_out': wash_out_notcancer[0],'cancer':0}
    return data,data2


def all_from_mask_contrast_img(data, path, img2_path_per, img15_path_per, dyn_image_per):

    dyn_image, affine = get_images(path, dyn_image_per, 0, 1)
    wash_in=np.zeros((dyn_image.shape[1],dyn_image.shape[2],dyn_image.shape[3]))
    wash_out=np.zeros((dyn_image.shape[1],dyn_image.shape[2],dyn_image.shape[3]))
    time = get_time()


    for i in range(50,150):
        print(i)
        for j in range(50,150):    
            for k in range(dyn_image.shape[3]):
                 TAC = dyn_image[0,i,j,k,:]

                 [wash_in[i,j,k],_],[ wash_out[i,j,k],_],TAC = get_wash_in_washout_for_TAC(TAC)

    _, affine = get_images(path, dyn_image_per, 0, 1)
    save_img(wash_in, affine, join(path,'wash_in.nii'))
    save_img(wash_out, affine, join(path,'wash_out.nii'))

def fit_model_for_minimaze(X,*params):
    K_tras, Ve= X
    time,blood,tac,ii=params
    Bj = np.asarray([])
  

    


    
    

    
    for i in ii:
##    for i in range(len(ii)):
     t=time[i]

     i = i + 1
     tua = time[0:i]

     c_p = blood[0:i]

     dtua = np.diff(np.insert(tua, 0, 0))

     y = np.exp(-(K_tras/Ve) * (t - tua)) * c_p

     Bj = np.append(Bj,np.sum(dtua*y))
    return   np.sum(np.power(K_tras * Bj -tac[ii],2))
##     return   np.sum(np.power(K_tras*  Bj -tac,2)) 

        

def fit_model_ii(data, K_tras, Ve):
 time = np.asarray(data[0])

 blood = np.asarray(data[1])

 Bj = np.asarray([])

 for i, t in enumerate(time):
     i = i + 1

     tua = time[0:i]

     c_p = blood[0:i]

     dtua = np.diff(np.insert(tua, 0, 0))

     y = K_tras *np.exp(-K_tras/Ve * (t - tua)) * c_p

     Bj = np.append(Bj,np.sum(dtua*y))

 return  Bj



def fit_model(data, K_tras, K_kep):
 time = np.asarray(data[0])

 blood = np.asarray(data[1])

 Bj = np.asarray([])

 for i, t in enumerate(time):
     i = i + 1

     tua = time[0:i]

     c_p = blood[0:i]

     dtua = np.diff(np.insert(tua, 0, 0))

     y = np.exp(-K_kep * (t - tua)) * c_p

     Bj = np.append(Bj,np.sum(dtua*y))

 return K_tras * Bj

def get_ks(time, aif, TAC,ii):
    x0=[0.2,0.2]
    time=np.asarray(time)

    params=  time,aif,TAC,ii
##    print(j)
    lw = [0,0]

    up = [2,1]
##    ret = optimize.minimize(fit_model_for_minimaze,x0,args=params, method='Powell', bounds=list(zip(lw, up)),options={'ftol': 0.002, 'xtol': 0.002})

##    ret = optimize.minimize(fit_model_for_minimaze,x0,args=params, bounds=list(zip(lw, up)))
##    bounds=list(zip(lw, up))s                        
    
    #,no_local_search=True
    ret =optimize.dual_annealing(fit_model_for_minimaze, bounds=list(zip(lw, up)),args=params)
####                                 ,initial_temp=1,restart_temp_ratio=0.8)#,no_local_search=True, args=params,x0=x0)
##    if ret.x[0]>2 or ret.x[1] >1 or ret.x[0]<0 or ret.x[1]<0:
####        ret.x=[0,0]

    return ret.x




def get_3_points(Ves,K_trass,ts,aif):
    #ii=[3, 9, 34]
    redgreenblue=np.ones([len(Ves),len(K_trass)])
    num=len(Ves)*len(K_trass)
    
    tac=np.zeros([len(Ves),len(K_trass),len(aif)])

    for l,Ve in enumerate(Ves):
                for k,K_tras in enumerate(K_trass):
                        tac[l,k,:]=fit_model([ts,aif], K_tras, K_tras/Ve)
    jj=np.argmax(aif[:20])
    ii=[jj,9, 34]
    for t in [0.2]:                  
        for i in range(jj + 1, 20):
            for j in range(34,30,-1):
                temp=np.zeros([len(Ves),len(K_trass)])+0.5
                for l,Ve in enumerate(Ves):
                    for k,K_tras in enumerate(K_trass):
                            c=tac[l,k,i]
                            d=tac[l,k,j]
                            if (d-c)/c>t:
                                
                                temp[l,k]=1
                            elif (d-c)/c<-t:
                                temp[l,k]=0

 
                temp_num=np.abs(np.sum(temp<0.5)-np.sum(temp>0.5))#/(np.abs(np.sum(temp<0.5)+np.sum(temp>0.5)))

                if  temp_num<num and (np.sum(temp<0.5)>0 and np.sum(temp>0.5)>0):
                    redgreenblue=temp
                    #i-5 0.7?
                    ii=[jj,i,j]
                    num=temp_num
                    if num==0:
                         plt.imsave(join(path, "cal"+ ".png"), redgreenblue)
                         break
                                 
                    
                    

                    #temp_num<num and
               
    print(ii)           
    plt.imsave(join(path, "cal"+ ".png"), redgreenblue)

    return ii
def wavelet_denoising(x, wavelet='db10'):
     
    coeff = pywt.wavedec(x, wavelet, mode="constant")
    sigma=stats.median_abs_deviation(np.asarray(coeff).flatten())/0.6745
    N =len(x.flatten())
    M=len(coeff[1:])
    #uthresh = sigma/stats.norm.cdf((1-0.05)/(2*M))
    uthresh= sigma*np.sqrt(2*np.log(N))
    #coeff[0] = pywt.threshold(coeff[0], value=uthresh, mode='hard')
    coeff[1:] = (pywt.threshold(i, value=   uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='constant')



def get_df_g():
    
    df_g=pd.read_excel('D:\\g.xlsx')
    group=df_g['g'].values.copy()



    group[group=='3+3']=0

    group[group=='3+4']=0

    group[group=='4+3']=1

    group[group=='4+4']=1

    group[group=='3+5']=1

    group[group=='5+3']=1

    group[group=='5+4']=1

    group[group=='4+5']=1

    group[group=='5+5']=1

    group[group==9]=1
    group[group==8]=1

    df_g['group']=group

    return df_g
    


mypath= 'C:\\Users\\liron\\Downloads\\copy'
#7
mypaths = [join(mypath, f) for f in listdir(mypath) if f.startswith('sub') ]
K_trass=np.arange(0.1, 2, 0.1)
Ves=np.arange(0.05, 1,0.05)#0.05:

ts=get_time()




data1 = {}

data2 = {}

data12 = {}

data22 = {}

aif=np.asarray([0]*35)
count=0
for path in mypaths:
   
    path = join(path,'DCE')
    img1_path_per = ['2deg','nii']

    img2_path_per =[ '15deg','nii']

    dyn_image_per = ['dyn','nii.g']
    #dynamicImaging = [join(path, f) for f in listdir(path) if  img2_path_per[0] in f   and img2_path_per[1] in f][:-3]
##
##    if len(dynamicImaging)==0:
##        continue 
    dyn_image, affine = get_images(path, dyn_image_per, 0, 1)
   

    aif_temp,j=get_tac_all_from_mask(path, img1_path_per, img2_path_per, dyn_image, ['blood_mask_DCE2.n','nii'])
    if np.max(aif_temp)>6:# and np.max(aif_temp)<20:#np.argmax(aif_temp)<20 and np.max(aif_temp)<5:
        i=np.argmax(aif_temp)-2
##        aif[3:35-max(0,i-3)]=aif[3:35-max(0,i-3)]+aif_temp[i:35-max(0,3-i)]
        aif=aif+aif_temp

        print(path)
        count=count+1
      
                
aif=aif/count       
plt.plot(aif)
plt.show()
######path=mypath
ii=get_3_points(Ves,K_trass,ts,aif)





t=np.cumsum([7.55]*35)/60

###
##A1,A2,T1,T2,sig1,sig2,alpha,beta,s,tua= 0.809, 0.330, 0.17046, 0.365 ,0.0563, 0.132, 1.050, 0.1685, 38.078, 0.483
##aif=A1/(sig1*np.sqrt(2*np.pi))*np.exp(-np.power((t-T1),2)/(2*np.power(sig1,2)))+alpha*np.exp(-beta*t)/(1+np.exp(-s*(t-tua)))\
##     +A2/(sig2*np.sqrt(2*np.pi))*np.exp(-np.power((t-T2),2)/(2*np.power(sig2,2)))+alpha*np.exp(-beta*t)/(1+np.exp(-s*(t-tua)))
##aif=np.asarray([0,0,0]+list(aif[:-3]))
######        
####mypaths = [join(mypath, f) for f in listdir(mypath) if f.startswith('sub') ]
##t=np.cumsum([7.55]*35)/60
##A1,A2,T1,T2,sig1,sig2,alpha,beta,s,tua= 0.809, 0.330, 0.17046, 0.365 ,0.0563, 0.132, 1.050, 0.1685, 38.078, 0.483
##aif=A1/(sig1*np.sqrt(2*np.pi))*np.exp(-np.power((t-T1),2)/(2*np.power(sig1,2)))+alpha*np.exp(-beta*t)/(1+np.exp(-s*(t-tua)))\
##     +A2/(sig2*np.sqrt(2*np.pi))*np.exp(-np.power((t-T2),2)/(2*np.power(sig2,2)))+alpha*np.exp(-beta*t)/(1+np.exp(-s*(t-tua)))
#ii=get_3_points(Ves,K_trass,ts,aif)

      

for path in mypaths:
##    if 'sub1' in path or 'sub17' in path:
##        continue
    root=path
    path = join(path,'DCE')

    img1_path_per = ['2deg','nii']

    img2_path_per =[ '15deg','nii']

    dyn_image_per = ['dyn','nii.g']
##
    ts=get_time()
    dynamicImaging = [join(path, f) for f in listdir(path) if  MASK_NOTCANCER[0] in f ]
    if len(dynamicImaging)<1:
        print("here")
        continue

    dyn_image, affine = get_images(path, dyn_image_per, 0, 1)#






##    data1,data12 = all_from_mask_contrast(data1,data12, path, img1_path_per, img2_path_per, dyn_image)


    data2,data22=all_from_mask_model(data2,data22,path,img1_path_per,img2_path_per,dyn_image,ii,aif)
    df = pd.DataFrame(data1)
    df=df.T

    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\h.csv')

    df = pd.DataFrame(data2)
    df=df.T
    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\h.csv')
