import numpy as np

import nibabel as nib

from os.path import join

from os import listdir

import pandas as pd

from matplotlib import pyplot as plt

from scipy import optimize
import HYPR_LR

from general_functions import *
from bilateral_filter import BilateralFilter



__win_x1, __win_y1, __win_z1 = 0, 0, 0  # 60, 40, 0

__win_x2, __win_y2, __win_z2 = 192, 192, 20  # 140, 1
0, 20
MASK_CANCER=['rcancer_di','nii']
MASK_NOTCANCER=['rnotcancer_dix','nii']
__COUNT1=0
__COUNT2=0
def save_2plot(path, x1, y1, x2, y2, title, x_label, y_label):

    fig, ax = plt.subplots()

    ax.plot(x1, y1, 'ro', x2, y2)

    ax.set_title(title)

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    fig.savefig(join(path, title + ".png"))
    plt.close('all')


def save_plot(path, x1, y1, title, x_label, y_label):

    fig, ax = plt.subplots()

    ax.plot(x1, y1)

    ax.set_title(title)

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    fig.savefig(join(path, title + ".png"))


def get_images(mypath, prefix, big_slice, end_slice):
    dynamicImaging = [join(mypath, f) for f in listdir(mypath) if  prefix[0] in f and prefix[1] in f]
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

    return (numerator1 / denominator1) * (numerator2 / denominator2)-sol


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

    return (numerator1 / denominator1) * (numerator2 / denominator2)-sol







def get_TAC_from_mask(images, mask_pre):

    mask, affine = get_images(path, mask_pre, 0, 1)
 

    images_temp=[(images[:,:,:,:,s]-images[:,:,:,:,0])/images[:,:,:,:,0] for s in range(images.shape[4])]
    
    images_temp = np.asarray(np.transpose(images_temp,(1,2,3,4,0)))
    images_temp[np.isnan(images_temp)] = 0
    images_temp[np.isinf(images_temp)] = 0

    TAC = np.asarray([np.mean(images_temp[:, :, :, :, i][mask>0]) for i in range(35)])
    j=np.sum(TAC[:6]<0.2)-1
##
##    images_temp=[(images[:,:,:,:,s]-images[:,:,:,:,j])/images[:,:,:,:,j] for s in range(images.shape[4])]
##    
##    images_temp = np.asarray(np.transpose(images_temp,(1,2,3,4,0)))
##    images_temp[np.isnan(images_temp)] = 0
##    images_temp[np.isinf(images_temp)] = 0
        
    


    TAC = np.asarray([np.mean(images[:, :, :, :, i][mask>0]) for i in range(35)])
    
    return TAC,j


def get_sigma_from_mask(images, mask_pre):

    mask, affine = get_images(path, mask_pre, 0, 1)

    sigma = np.asarray([np.mean(images[:, :, :, :, i][mask>0]) for i in range(35)])

   
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
        return np.cumsum([7.55/60] * 35)-3.8



def get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image_per, mask_per,):
##    img2, affine = get_images(path, img2_path_per, 0, 1)
##
##    img15, affine = get_images(path, img15_path_per, 0, 1)
##
##    s2 = get_region_from_mask(img2, mask_per)
##
##    s15 = get_region_from_mask(img15, mask_per)
##
##    t1_0 = optimize.fsolve(get_t0,[0.8],args=([2 * 0.017, 15 * 0.017,s2 / s15]),factor=0.8)
##
##    dyn_image, affine = get_images(path, dyn_image_per, 0, 1)
####
    dyn_image, affine = get_images(path, dyn_image_per, 0, 1)

    t1_obs = np.asarray([])

    s_dyn,j = get_TAC_from_mask(dyn_image, mask_per)

    s_dyn = np.asarray(s_dyn)
##
##    for i in range(dyn_image.shape[4]):
##        popt = optimize.fsolve(get_t_obs,[2], args=([15 * 0.017,2 * 0.017 , t1_0,s_dyn[i] / s2]),factor=0.5)
##
##        t1_obs = np.append(t1_obs, popt)
##    
##
##    TAC = 1 / (4.5) * (1 / t1_obs - 1 / t1_0)
##
##    TAC[TAC<0]=0

    return s_dyn,j



def get_tac_all_from_mask_img(path, s2, s15, s_dyn):



    t1_0, pcov = optimize.curve_fit(get_t0, [2 * 0.0174533, 15 * 0.0174533], [s2 / s15],bounds=[0,2])

    t1_0 = t1_0[0]


    t1_obs = np.asarray([])

    for i in range(35):
        popt, pcov = optimize.curve_fit(get_t_obs, [15 * 0.0174533, 15 * 0.0174533, t1_0,s_dyn[i] / s15]

                                        ,bounds=[0,2])

        t1_obs = np.append(t1_obs, popt[0])

    TAC = 1 / (4.5*60) * (1 / t1_obs - 1 / t1_0)

    return TAC


##def get_ks(time, aif, TAC,sigma):
##    
##    popt, pcov = optimize.curve_fit(fit_model, [time, aif],TAC, maxfev=80000,p0=[0.2,0.5])
##    print(popt)
##
##
##    return popt
##
##def fit_model_h(time, K_in, K_out,k12,k21):
##    
##
##
## return 1+a


from sklearn.metrics import r2_score



                                      
def all_from_mask_model(data,data2, path, img2_path_per, img15_path_per, dyn_image_per):

    TAC_cancer,j_cancer = get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image_per, MASK_CANCER)

    TAC_notcancer,j_notcancer = get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image_per, MASK_NOTCANCER)


    print(j_notcancer)
    time = get_time()
    #aif,_=get_tac_all_from_mask(path, img2_path_per, img15_path_per, dyn_image_per, ['rblood_mask.n','nii'])
    i=7
    K_tras_cancer, Ve_cancer = get_ks(time, aif, TAC_cancer,[0,6,34])
    K_kep_cancer=K_tras_cancer/Ve_cancer
    K_tras_notcancer, Ve_notcancer = get_ks(time, aif, TAC_notcancer,[0,7,34])
    K_kep_notcancer=K_tras_notcancer/Ve_notcancer

    
    
    save_plot(path, time, aif ,'aif', '', '')
  

    save_2plot(path, time, TAC_cancer, time, fit_model([time, aif], K_tras_cancer, K_kep_cancer)
               
               ,'K_tras_cancer ' + str(K_tras_cancer) + 'K_kep_cancer ' + str(K_kep_cancer), '', '')

    save_2plot(path, time, TAC_notcancer, time, fit_model([time, aif], K_tras_notcancer, K_kep_notcancer), 
            'K_tras_notcancer ' + str(K_tras_notcancer) + 'K_kep_notcancer ' + str(K_kep_notcancer), '', '')

    data[path.split('\\')[-2]] = {'K_tras_cancer': K_tras_cancer, 'Ve_cancer': Ve_cancer,'K_kep_cancer':K_kep_cancer,
                              'K_tras_notcancer': K_tras_notcancer, 'Ve_notcancer':Ve_notcancer,'K_kep_mptcancer': K_kep_notcancer}

    
    return data,[]


def all_from_mask_model_img(data, path, img2_path_per, img15_path_per, dyn_image_per):

    dyn_img, affine = get_images(path, dyn_image_per, 0, 1)
    img2, affine = get_images(path, img2_path_per, 0, 1)
    img15, affine = get_images(path, img15_path_per, 0, 1)
    K_tras=np.zeros((dyn_img.shape[1],dyn_img.shape[2],dyn_img.shape[3]))
    K_kep=np.zeros((dyn_img.shape[1],dyn_img.shape[2],dyn_img.shape[3]))
    #wash_in,affine= get_images(path, 'wash_in.nii', 0, 1)
    mask, affine = get_images(path, MASK_CANCER, 0, 1)
    mask2, affine = get_images(path, MASK_NOTCANCER, 0, 1)
    mask=mask+mask2
    #th=np.sort(wash_in.flatten())
    #th=th[-int(len(th)/100)*10]
    #print(th)

    time = get_time()
    _,x_size,y_size,z_size=img2.shape
    for i in range(x_size):
        print(i)
        for j in range(y_size):    
            for k in range(z_size):
                if mask[0,i,j,k]>0  and img2[0,i,j,k]>0 and img15[0,i,j,k]>0:
                    TAC=get_tac_all_from_mask_img(path, img2[0,i,j,k], img15[0,i,j,k], dyn_img[0,i,j,k,:])
                    K_tras[i,j,k], K_kep[i,j,k] = get_ks(time, aif, TAC)


        save_img(K_tras, affine, join(path,'K_tras.nii'))
        save_img(K_kep, affine, join(path,'K_kep.nii'))

    return data




def all_from_mask_contrast(data,data2, path, img2_path_per, img15_path_per, dyn_image_per):

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
    K_tras, Ve = X
    time,blood,tac,ii=params
    Bj = np.asarray([])
    


    
    

    
    for i, t in enumerate(time):
     i = i + 1

     tua = time[0:i]

     c_p = blood[0:i]

     dtua = np.diff(np.insert(tua, 0, 0))

     y = np.exp(-(K_tras/Ve) * (t - tua)) * c_p

     Bj = np.append(Bj, np.trapz(y , x=tua))
    return   np.sum(np.power(K_tras * Bj[ii] -tac[ii],2)) 
        



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

     Bj = np.append(Bj, np.trapz(y , x=tua))

 return K_tras * Bj

def get_ks(time, aif, TAC,j):

    x0=[0.2,0.04]
    time=np.asarray(time)
    params=  time,aif,TAC,j
##    print(j)
    lw = [0,0]

    up = [2,1] 
    ret = optimize.minimize(fit_model_for_minimaze,x0,args=params, bounds=list(zip(lw, up)),tol=2e-2)
                            


    ##    ret =optimize.dual_annealing(fit_model_for_minimaze, bounds=list(zip(lw, up)), args=params,x0=[0.2,0.5],initial_temp=50000)

    return ret.x

##
mypaths = [
        'C:\\Users\\liron\\Downloads\\copy\\sub12',
        'C:\\Users\\liron\\Downloads\\copy\\sub17',

        'C:\\Users\\liron\\Downloads\\copy\\sub14',
        'C:\\Users\\liron\\Downloads\\copy\\sub13',
        'C:\\Users\\liron\\Downloads\\copy\\sub16',
        'C:\\Users\\liron\\Downloads\\copy\\sub19',
        
        'C:\\Users\\liron\\Downloads\\copy\\sub15',
        'C:\\Users\\liron\\Downloads\\copy\\sub20',
        'C:\\Users\\liron\\Downloads\\copy\\sub87',
        'C:\\Users\\liron\\Downloads\\copy\\sub86',
        'C:\\Users\\liron\\Downloads\\copy\\sub85',
        'C:\\Users\\liron\\Downloads\\copy\\sub82',
        'C:\\Users\\liron\\Downloads\\copy\\sub8',
        'C:\\Users\\liron\\Downloads\\copy\\sub3',
        'C:\\Users\\liron\\Downloads\\copy\\sub5',
        'C:\\Users\\liron\\Downloads\\copy\\sub6',
        'C:\\Users\\liron\\Downloads\\copy\\sub7',
        'C:\\Users\\liron\\Downloads\\copy\\sub9',
       
        'C:\\Users\\liron\\Downloads\\copy\\sub21',
        'C:\\Users\\liron\\Downloads\\copy\\sub24',
        'C:\\Users\\liron\\Downloads\\copy\\sub26',
        'C:\\Users\\liron\\Downloads\\copy\\sub27',
        'C:\\Users\\liron\\Downloads\\copy\\sub29',
        'C:\\Users\\liron\\Downloads\\copy\\sub25',
        
        'C:\\Users\\liron\\Downloads\\copy\\sub30',
        'C:\\Users\\liron\\Downloads\\copy\\sub42',
        'C:\\Users\\liron\\Downloads\\copy\\sub32',
        'C:\\Users\\liron\\Downloads\\copy\\sub34',
        'C:\\Users\\liron\\Downloads\\copy\\sub45',
        'C:\\Users\\liron\\Downloads\\copy\\sub36',
        'C:\\Users\\liron\\Downloads\\copy\\sub37',
        'C:\\Users\\liron\\Downloads\\copy\\sub51',

        'C:\\Users\\liron\\Downloads\\copy\\sub49',
        #'C:\\Users\\liron\\Downloads\\copy\\sub48',

        'C:\\Users\\liron\\Downloads\\copy\\sub46',
        'C:\\Users\\liron\\Downloads\\copy\\sub43',
        'C:\\Users\\liron\\Downloads\\copy\\sub40',
        'C:\\Users\\liron\\Downloads\\copy\\sub52',
        'C:\\Users\\liron\\Downloads\\copy\\sub53',
        'C:\\Users\\liron\\Downloads\\copy\\sub56',

        'C:\\Users\\liron\\Downloads\\copy\\sub58',
        'C:\\Users\\liron\\Downloads\\copy\\sub60',]
    


mypath= 'C:\\Users\\liron\\Downloads\\copy\\'
##
mypaths = [join(mypath, f) for f in listdir(mypath) if f.startswith('sub') ]



data1 = {}

data2 = {}

data12 = {}

data22 = {}
##
##
##count=0
##
##aif=np.zeros(35)
##for path in mypaths[:102]:
##    if not('sub405'  in path or 'sub52'  in path or 'sub103'  in path):
##        path = join(path,'DCE')
##
##        img1_path_per = ['2deg','nii']
##        img2_path_per =[ '15deg','nii']
##        num=1
##        dyn_image_per = ['dyn','nii.g']
##        aif_indiv,_=get_tac_all_from_mask(path, img1_path_per, img2_path_per, dyn_image_per, ['blood_mask.n','nii'])
##        aif=aif_indiv+aif
##        ii=np.argmax(aif_indiv)-3
##        if np.max(aif_indiv)>0:
##            aif=aif_indiv+aif
##            count=count+1
######
######        
######            if num>ii :
######                aif[num:]=aif_indiv[ii:35-(num-ii)]+aif[num:]
######                count=count+1
######            elif  num<=ii :
######                aif[num:35-(ii-num)]=aif_indiv[ii:]+aif[num:35-(ii-num)]
######                count=count+1
########        else:
########            print(path)
######
##########
##aif=aif/count
##plt.plot(aif)
##plt.show()
##print(count)

########
####        
##mypaths = [join(mypath, f) for f in listdir(mypath) if f.startswith('sub') ]
##t=np.cumsum([7.55]*35)/60
##A1,A2,T1,T2,sig1,sig2,alpha,beta,s,tua= 0.809, 0.330, 0.17046, 0.365 ,0.0563, 0.132, 1.050, 0.1685, 38.078, 0.483
##aif=A1/(sig1*np.sqrt(2*np.pi))*np.exp(-np.power((t-T1),2)/(2*np.power(sig1,2)))+alpha*np.exp(-beta*t)/(1+np.exp(-s*(t-tua)))\
##     +A2/(sig2*np.sqrt(2*np.pi))*np.exp(-np.power((t-T2),2)/(2*np.power(sig2,2)))+alpha*np.exp(-beta*t)/(1+np.exp(-s*(t-tua)))

##aif=[0.10991932, 0.08203243, 0.09734448, 0.10779402, 1.16233783,
##       3.61822105, 3.04398219, 2.48681683, 2.36565569, 2.21757448,
##       2.07370103, 1.97081596, 1.94867821, 1.84874058, 1.81580038,
##       1.74718922, 1.7055877 , 1.67791667, 1.56419912, 1.56295906,
##       1.51178126, 1.4739364 , 1.44862453, 1.38803217, 1.36287275,
##       1.31160627, 1.32965315, 1.31010368, 1.28244435, 1.27452811,
##       1.23535341, 1.21521877, 1.21132078, 1.19098284, 1.18442032]


aif=[ 673.25788279,  649.25854687,  659.68569554,  666.09942657,
        962.14054066, 1482.59617178, 1455.95400036, 1386.33422933,
       1371.33129932, 1345.15398688, 1321.04962117, 1300.47219515,
       1292.66798652, 1274.20894317, 1266.62160917, 1249.96729837,
       1238.55218426, 1234.90717205, 1205.82370159, 1203.07747032,
       1189.2792679 , 1183.47259743, 1172.48103562, 1155.78870847,
       1151.11243592, 1138.61233305, 1142.52834125, 1139.52654571,
       1127.50072255, 1126.80714442, 1116.74799192, 1112.42119415,
       1106.826103  , 1103.98055772, 1099.79680069]


for path in mypaths:
    path = join(path,'DCE')
    img1_path_per = ['2deg','nii']

    img2_path_per =[ '15deg','nii']


    dyn_image_per = ['dyn','nii.g']



    print(path)

    img1_path_per = ['2deg','nii']

    img2_path_per =[ '15deg','nii']

##

    
##    dyn_img, affine = get_images(path, dyn_image_per, 0, 1)
##    a=BilateralFilter()
##    a.denoise(path,dyn_img,affine)
##    dyn_image_per = ['bf','nii']
##
##
##    data1 = all_from_mask_contrast_img(data1, path, img1_path_per, img2_path_per, dyn_image_per) 
##    data2=all_from_mask_model_img(data2,path,img1_path_per,img2_path_per,dyn_image_per)
##    df = pd.DataFrame(data2)
##    df=df.T
##    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\all_from_mask_model.csv')

     




    data1,data12 = all_from_mask_contrast(data1,data12, path, img1_path_per, img2_path_per, dyn_image_per)

   
    data2,data22=all_from_mask_model(data2,data22,path,img1_path_per,img2_path_per,dyn_image_per)
  
    df = pd.DataFrame(data1)
    df=df.T

    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\all_from_mask_contrastdyns_h6.csv')

    df = pd.DataFrame(data2)
    df=df.T
    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\all_from_mask_modeldyns_h6.csv')


    df = pd.DataFrame(data12)
    df=df.T

    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\contrast_for_fit.csv')

    df = pd.DataFrame(data22)
    df=df.T
    df.to_csv('C:\\Users\\liron\\Downloads\\copy\\model_for_fit.csv')
    __COUNT1= __COUNT1 +2
    __COUNT2= __COUNT2 +2

