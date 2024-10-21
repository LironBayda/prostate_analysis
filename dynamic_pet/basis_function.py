# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:17:15 2021

@author: assuta
"""
from os import listdir
from os.path import isfile, join
from os import system
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from get_data_from_dicoms import get_variables_from_dicom
import cupy as cp

# root_dir='D:\Pet_dynamic18032021\_DELAY_157_10X30_6X300_1X263_PRR_AC_IMAGES\check'
# folders = [join(root_dir, f) for f in listdir(root_dir)]

################################################################################################################################################
# 10,15
# 10,15,14 'D:\data_from_pet-MRI2\sub-5','D:\data_from_pet-MRI2\sub-29' 'D:\data_from_pet-MRI2\sub-27',\
#       'D:\data_from_pet-MRI2\sub-17' ,'D:\data_from_pet-MRI2\sub-24','D:\data_from_pet-MRI2\sub-37'\
#      ,'D:\data_from_pet-MRI2\sub-19' ,'D:\data_from_pet-MRI2\sub-28','D:\data_from_pet-MRI2\sub-32'\
# 'D:\data_from_pet-MRI2\sub-36','D:\data_from_pet-MRI2\sub-38','D:\data_from_pet-MRI2\sub-39','D:\data_from_pet-MRI2\sub-40'

folders = ['D:\\data_from_pet-MRI2\\sub-12', 'D:\\data_from_pet-MRI2\\sub-15', 'D:\\data_from_pet-MRI2\\sub-17'
    , 'D:\\data_from_pet-MRI2\\sub-21', 'D:\\data_from_pet-MRI2\\sub-24', 'D:\\data_from_pet-MRI2\\sub-32'
    , 'D:\\data_from_pet-MRI2\\sub-37', 'D:\\data_from_pet-MRI2\\sub-38', 'D:\\data_from_pet-MRI2\\sub-40']

folders = ['D:\data_from_disk_3/inList/sub-17withrawdata']


def STCCorrection(folders):
    for mypath in folders:
        mypath = join(mypath, 'dynPET')
        files = [f for f in listdir(mypath) if f.startswith('r20')]
        # E:\PETPVC_bulid_new\src\Debug>petpvc -i
        # E:\_10X30_6X300_1X420_PRR_AC_IMAGES_40001\GLYO_4002\r20181025_120001s40002a000_17
        # -o E:\_10X30_6X300_1X420_PRR_AC_IMAGES_40001\GLYO_4002\STC_r0181025_120001s40002a000_17.nii
        # -m E:\_10X30_6X300_1X420_PRR_AC_IMAGES_40001\GLYO_4002\cortid_mask -p "STC" -x 4.6 -y 4.6 -z 4.6

        for i in range(24,25):
            cmd = 'D: && cd D:\PETPVC_bulid_new\src\Debug && petpvc -i {fileIn} -o {fileOut} -m {mask} -p STC -x 4.6 -y 4.6 -z 4.6' \
                .format(fileIn=join(mypath, files[i]), fileOut=join(mypath, "STC" + files[i]),
                        mask=join(mypath, "artery_mask.nii"))
            system(cmd)
            print(cmd);


# STCCorrection(folders)

############################################################################################################################################3

"""
>>> import nipype.interfaces.spm as spm
    matlab_cmd = 'C:\MATLAB\spm12 D:\Program Files\MATLAB\R2021a\bin'
   spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

>>> smooth = spm.Smooth()
>>> smooth.inputs.in_files = 'functional.nii'
>>> smooth.inputs.fwhm = [4, 4, 4]
>>> smooth.run() 
    '''
    Parametric Imaging of Ligand-Receptor Binding in PET Using
    a Simplified Reference Region Model

    https://pdf.sciencedirectassets.com/272508/1-s2.0-S1053811900X0047X/1-s2.0-S1053811997903037/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHQG0K3l96SWDgWF4ksNo3bElMyI5VzLfTsuoDyJkxhmAiEAtmydLPuxj98s4tPGVpal7laHAiz4LLCMjEzs1Br5LAIqgwQIp%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDE%2BvOdgI1xqjZVoHmSrXA4IQiDY7PhP6ycV7DBZIB9DygtytIRWIjb7nxTblQ6ZaYBE%2FASY9rebttHbU6ymK29vJSS%2FTP6Oy1YvnTF5AzDX2f7XGZyJHgY7Yk4f7v5LJnWuuV7YFXNR0V8pzyEt8hrBVNFAzGWoG%2B38LixzwhzGV1vr5p5%2BC8ifSlo20j%2Be%2BxV2098ulksu28O0xHKQsUnIOMIGx%2Fpd0CYT%2Bx8jJOKralymEEISnwPWuVbSppBYFINeH3K5RSXlCiPyNRIh%2Fe2uqRQwGkxAM45jaAzf7XLjwT9CUB%2BVDY%2B2djfJzPgBOg7IcIykTt90%2BJlRPut5FsFCPQK52Q9KCJkoJtBeJZrhJ2fWav%2FRgnDTu7%2BtJLgEpFn%2Bhee5Nz4QHC5helFblnFPD2SZ6BP27rHP2X49KmK9K4qBzOkUXgcODgI1MSNO1Y%2FFXm4gEHSLtMGrwBVCsxEyY08ApsmWyni1Fq2TMg1Uc93QfLO4p4hHO%2FaI6RzcaVTvgQj02PDwBqY%2BNV%2FbrKIz9Ba36QJOIHRa2DmHz7LSCbDP%2BAUqYDkKqC6%2FxcosgFBmfKDvUuDgcjatOjeAJDrYLIZSf%2BYbK%2B9UjdveloSBSGxTS3R5ffS8M56cRbUuDfv%2BoJ4EWyTCLtJ%2BMBjqlAYu8iWgVyBHU%2FyUivSM%2Ba1V64otuXKiGxikd%2FS7y1PNfMESdFsOz8hhtcBGyzBvUjLsdreB7nVSTPjij8PB7MYVtPEZwhkRnCd89KSvyPRizEFx%2BLFNGbBKZI8GqWswexKeSbmVD%2FlrotgCW50UWTMZTSMGZXkzlls5pLrqVEVYskkjo7m5px0Pd6MaIp8nBJCQ8oefiyEKyEIp5SV3fcz79ASuo8w%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211107T150055Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTY6TUYPVHT%2F20211107%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=573b93e0d78ce126ea5b9badf6991b9d00db48a7d2a22f30cb02b4c67d8147be&hash=4cdc474df34ccc3f8f4fdcc7e39607fcc5d65e93641babccb9c217dcb76dd0a0&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1053811997903037&tid=spdf-0fe6c116-3fea-48cf-9942-ad9e7e8ae33e&sid=ba6429ff1eb34048520bb8a54ce3f906e48bgxrqb&type=client

    Kinetic modelling using basis functions derived from two-tissue compartmental models with a plasma input function: General principle and application to [18F]fluorodeoxyglucose positron emission tomography
    https://www.sciencedirect.com/science/article/pii/S1053811910001813#fd4


    Two-Tissue Compartment Model with Basis Functions
    http://doc35.pmod.com/pxmod/pmclass.lib.pmod.models.PM2TCBFM.html



    '''
 """
__x1 = 60
__x2 = 124
__y1 = 40
__y2 = 72
__A_for_All_B = cp.asarray([])
__w = cp.asarray([])
__W = cp.asarray([])
__M = cp.asarray([])


def get_images(mypath, prefix, big_slice, end_slice):
    dynamicImaging = [join(mypath, f) for f in listdir(mypath) if f.startswith(prefix)]
    images = [nib.load(path).get_fdata() for path in dynamicImaging[big_slice:end_slice]]
    images = cp.asarray(np.asarray(images))

    return images


def save_images(mypath, prefixI, prefixK, K1, K2, K3, K4, V):
    dynamicImaging = [join(mypath, f) for f in listdir(mypath) if f.startswith(prefixI)]
    firstimageaffine = nib.load(dynamicImaging[0]).affine
    array_img = nib.Nifti1Image(K1.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_K1.nii'))
    array_img = nib.Nifti1Image(K2.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_K2.nii'))
    array_img = nib.Nifti1Image(K3.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_K3.nii'))
    array_img = nib.Nifti1Image(K4.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_K4.nii'))
    array_img = nib.Nifti1Image(V.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_V.nii'))
    array_img = nib.Nifti1Image(K1 / K2.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_K1_K2.nii'))
    array_img = nib.Nifti1Image((K1 * K3) / (K2 + K3).astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_Ki.nii'))
    array_img = nib.Nifti1Image(K3 / K4.astype(np.float64), firstimageaffine)
    nib.save(array_img, join(mypath, prefixK + '_K3_K4.nii'))

    return images


def get_carotid_mask(mypath):
    return cp.asarray(nib.load(join(mypath, "artery_mask.nii")).get_fdata())


def get_weights(slicesDuration, images, NumberOfTimeSlices, big_slice, end_slice, DoseCalibrationFactor):
    # weightes
    # starat with 1 because the first one is zero
    #   w=[slicesDuration[i]*slicesDuration[i]/(np.mean(voxelSize[0]*voxelSize[1]*voxelSize[2]*images[i]*slicesDuration[i])) for i in  range(1,NumberOfTimeSlices) ]
    w = [cp.power(slicesDuration[i], 2) / (cp.sum(DoseCalibrationFactor * images[i])) for i in
         range(0, end_slice - big_slice)]
    # w.insert(0,0)
    #   w=[1 for i in  range(1,NumberOfTimeSlices) ]
    w = cp.asarray(w)
    W = cp.diag(cp.sqrt(w))
    w = cp.asarray([w])
    return w, W


def get_alphas(alpha1Max, alpha1Mim, alpha2Max, alpha2Mim, numberOfBasisFunction):
    step = (alpha1Max - alpha1Mim) / (numberOfBasisFunction - 1)
    alpha1 = [alpha1Mim + num * step for num in range(0, numberOfBasisFunction)]
    step = (alpha2Max - alpha2Mim) / (numberOfBasisFunction - 1)

    alpha2 = [alpha2Mim + num * step for num in range(0, numberOfBasisFunction)]
    return cp.asarray(alpha1), cp.asarray(alpha2)


def get_c_P(carotid_mask, big_slice, end_slice, images):
    images = get_images(mypath, "STCr20", big_slice, end_slice)
    c_p = cp.asarray([cp.mean(image[carotid_mask == 1]) for image in images])
    return cp.asarray(c_p)


def get_TACs_for_images(images, numSlice):
    time_slice, sizeX, sizeY, sizeZ = images.shape
    # images=images[:,int(2*sizeX/5):int(sizeX-2*sizeX/5),int(2*sizeX/5):int(sizeY-2*sizeY/5),numSlice]
    # time_slice,sizeX,sizeY=images.shape
    images = images[:, __x1:__x2, __y1:__y2, numSlice]

    TACs = cp.reshape(images, [time_slice, (__x1 - __x2) * (__y1 - __y2)])
    # pl
    return TACs, sizeX, sizeY


def convolve(alpha, blood, end_time, slicesDuration):
    results = [];
    blood[blood < 0] = 0;
    blood = np.insert(blood,0,0)
    slicesDuration=np.insert(slicesDuration,0,end_time[0]-slicesDuration[1])
    dt = np.min(slicesDuration)
    end_time=np.insert(end_time,0,end_time[0] - dt)
    step=int(dt*60)
    for i in range(2,len(end_time)+1):
        tau = np.asarray(range(0,int(60*end_time[i-1])+step,step))/60
        num_of_point =slicesDuration*60/ step
        num_of_point=np.asarray([int(x) for x in num_of_point])
        c_p = [0]
        for j in range(i):
            for ll in range(int(num_of_point[j])):
                c_p.append(blood[j]);

        results.append(np.sum((np.exp(-alpha * tau) * np.flip(c_p)) * dt));

    return results


def get_M_and_A_for_All_B(numberOfBasisFunction, alpha1, alpha2, slicesDuration, mid, c_p):
    M = []
    A_for_All_B = []
    C_B = c_p.get()
    alphas = []
    index = []
    alpha1 = alpha1.get()
    alpha2 = alpha2.get()

    [[index.append((j1, j2)), alphas.append((alpha1[j1], alpha2[j2]))] for j1 in range(len(alpha1)) for j2 in
     range(len(alpha2))]
    slicesDuration = slicesDuration.get()
    mid = mid.get()
    c_p = c_p.get()

    for x, y in alphas:
        B1j = convolve(x, c_p, mid, slicesDuration)
        B2j = convolve(y, c_p, mid, slicesDuration)
        A = np.concatenate((B1j, B2j, C_B), axis=0)
        A = np.reshape(A, (len(C_B), 3))

        #   A=A.reshape(len(B1j),2)

        A_for_All_B.append(A)
        Q, R = np.linalg.qr(np.dot(__W.get(), A))
        M.append(np.dot(np.linalg.inv(R), Q.T))
    return cp.asarray(A_for_All_B), cp.asarray(M), cp.asarray(index)


# RSS_lock = threading.Lock()


def A_for_All_B_dot_sol(solForSingleVoxel, TAC, constrainSingle, i):
    print(i)

    solForSingleVoxel = cp.transpose(cp.array([solForSingleVoxel]), (1, 2, 0))
    C_T = cp.reshape(cp.matmul(__A_for_All_B, solForSingleVoxel), (__A_for_All_B.shape[0:2]))
    RSS = cp.sum(
        cp.dot(cp.ones((C_T.shape[0], 1)), __w) * cp.power(C_T - cp.dot(cp.ones((C_T.shape[0], 1)), cp.asarray([TAC])),
                                                           2),
        axis=1)
    RSS[constrainSingle == False] = cp.inf
    i_min_single = cp.argmin(RSS)
    return i, i_min_single, RSS[i_min_single]


def voxelWiseCalculation(TACs, c_p, alpha1, alpha2, index):
    #thr = TACs[-1, :]
    #thr = thr[thr != 0]
    #thr = cp.sort(thr)[int((__x2 - __x1) * (__x2 - __x1) / 100) * 20]
    thr=1
    indexTAC = cp.asarray(cp.where(TACs[-1, :] > thr)).T
    TACsbigthenth = cp.asarray(TACs[:, TACs[-1, :] > thr])
    alpha2 = np.array(alpha2.get())
    alpha1 = np.array(alpha1.get())
    index = index.get()
    sol = [np.dot(__M[ii].get(), np.dot(__W.get(), TACsbigthenth.get())) for ii in range(len(alpha1) * len(alpha2))]
    sol = np.transpose(np.array(sol), (0, 2, 1))
    theta1 = np.asarray(sol[:, :, 0])
    theta2 = np.asarray(sol[:, :, 1])
    v = np.asarray(sol[:, :, 1])
    K1_temp = (theta1 + theta2) / (1 + v)
    print(1)
    alpha1_sol = np.ones((theta1.shape[1], theta1.shape[0])) * alpha1[index[:, 0]]
    alpha2_sol = np.ones((theta1.shape[1], theta1.shape[0])) * alpha2[index[:, 1]]
    K2_temp = (theta1 * alpha1_sol.T + theta2 * alpha2_sol.T) / (theta1 + theta2)
    K4_temp = (alpha1_sol.T * alpha2_sol.T) / K2_temp
    K3_temp = (alpha1_sol.T + alpha2_sol.T) - K2_temp - K4_temp
    constrain = (K1_temp > 0) * (K2_temp > 0) * (K3_temp > 0) * (K4_temp >= 0) * (v >= 0)
    print(2)
    TACNum = cp.asarray([], dtype=cp.int64)
    i_min = cp.asarray([], dtype=cp.int64)
    RSS = cp.asarray([])

    constrain_sum_0 = np.sum(constrain, axis=0)
    print(sol.shape)
    for i in range(sol.shape[1]):
        if constrain_sum_0[i]:
            i, i_min_single, RSS_i_min_single = A_for_All_B_dot_sol(cp.asarray(sol[:, i, :]),
                                                                    cp.array(TACsbigthenth[:, i]),
                                                                    cp.asarray(constrain[:, i]), i)
            TACNum = cp.append(TACNum, i)
            i_min = cp.append(i_min, i_min_single)
            RSS = cp.append(RSS, RSS_i_min_single)

    print(TACsbigthenth.shape[1])
    print(TACNum)
    i_min = i_min.get()
    TACNum = TACNum.get()
    return tuple(indexTAC[TACNum].get()), [v[i_min, TACNum], K1_temp[i_min, TACNum], K2_temp[i_min, TACNum],
                                           K3_temp[i_min, TACNum], K4_temp[i_min, TACNum], RSS]


RSS = []

# STCCorrection(folders)
# 'D:\data_from_pet-MRI2\sub-14\dynPET' 'D:\data_from_pet-MRI2\sub-21\dynPET','D:\data_from_pet-MRI2\sub-3\dynPET'\
#  ,'D:\data_from_pet-MRI2\sub-10\dynPET','D:\data_from_pet-MRI2\sub-15\dynPET',
if __name__ == '__main__':
    # 5,17,19
    folders = ['D:\\data_from_disk_3\\inList\\sub-17withrawdata']
    #         'D:\\data_from_pet-MRI2\\sub-15', 'D:\\data_from_pet-MRI2\\sub-12', 'D:\\data_from_pet-MRI2\\sub-17',
    #       'D:\\data_from_pet-MRI2\\sub-21', 'D:\\data_from_pet-MRI2\\sub-24', 'D:\\data_from_pet-MRI2\\sub-32',
    #        'D:\\data_from_pet-MRI2\\sub-37', 'D:\\data_from_pet-MRI2\\sub-38', 'D:\\data_from_pet-MRI2\\sub-40',
    #        ]

    # STCCorrection(folders)
    for mypath in ['D:\\data_from_disk_3\\inList\\sub-17withrawdata']:  # folders:
        # set importent variables
        print('start main')

        mypath = mypath + '//dynPET'
        alpha1Mim, alpha1Max, alpha2Mim, alpha2Max = 0.005, 0.05, 0.2, 2
        numberOfBasisFunction = 100
        # mypath='D:\epilipsy\SHAY_DAFNA_335258877\_10X30_6X300_2X420_1X660_PRR_AC_IMAGES_40001'
        big_slice, end_slice = 0, 24
        savePerfix = "oneSlice"

        # pipeline
        print('get_images')
        NumberOfSlices, NumberOfTimeSlices, mid, slicesDuration, DoseCalibrationFactor = get_variables_from_dicom(
            mypath, big_slice, end_slice)
        mid = cp.asarray(mid)
        slicesDuration = cp.asarray(slicesDuration)

        images = get_images(mypath, "r20", big_slice, end_slice)
        # images=ndimage.gaussian_filter(images,1)
        carotid_mask = get_carotid_mask(mypath)

        __w, __W = get_weights(slicesDuration, images, NumberOfTimeSlices, big_slice, end_slice, DoseCalibrationFactor)

        alpha1, alpha2 = get_alphas(alpha1Max, alpha1Mim, alpha2Max, alpha2Mim, numberOfBasisFunction)

        if alpha1Mim + alpha1Max == 0:
            alpha1 = [0]
        c_p = get_c_P(carotid_mask, big_slice, end_slice, images)

        # for slicenum in range(images[0].shape[2]):
        # oneSliceOfimage= get_oneSliceOfimage(images,slicenum)

        print('get_M_and_A_for_All_B')

        __A_for_All_B, __M, index = get_M_and_A_for_All_B(numberOfBasisFunction, alpha1, alpha2, slicesDuration, mid,
                                                          c_p)

        TACs, sizeX, sizeY = get_TACs_for_images(images, 0)
        K1 = np.zeros([sizeX, sizeY, images.shape[3]])
        K2 = np.zeros([sizeX, sizeY, images.shape[3]])
        K3 = np.zeros([sizeX, sizeY, images.shape[3]])
        K4 = np.zeros([sizeX, sizeY, images.shape[3]])

        V = np.zeros([sizeX, sizeY, images.shape[3]])

        K1_window = np.zeros([(__x2 - __x1) * (__y2 - __y1), images.shape[3]])
        K2_window = np.zeros([(__x2 - __x1) * (__y2 - __y1), images.shape[3]])
        K3_window = np.zeros([(__x2 - __x1) * (__y2 - __y1), images.shape[3]])
        K4_window = np.zeros([(__x2 - __x1) * (__y2 - __y1), images.shape[3]])

        V_window = np.zeros([(__x2 - __x1) * (__y2 - __y1), images.shape[3]])
        for slicenum in range(30, 94):
            # slicenum=70

            print('get_TACs_for_single_image' + str(slicenum))
            TACs, sizeX, sizeY = get_TACs_for_images(images, slicenum)

            print('voxelWiseCalculation' + str(slicenum))
            # results=[]
            # indexTAC,executors_list
            indexTAC, results = voxelWiseCalculation(TACs, c_p, alpha1, alpha2, index)
            # for output in executors_list:
            #   results.append(output.result())
            indexTAC = np.asarray([indexTAC[i][0] for i in range(len(indexTAC))])
            print(indexTAC.shape)
            print(results[0].shape)

            print('KCalculation' + str(slicenum))
            V_window[indexTAC, slicenum] = results[0]
            K1_window[indexTAC, slicenum] = results[1]
            K2_window[indexTAC, slicenum] = results[2]
            K3_window[indexTAC, slicenum] = results[3]
            K4_window[indexTAC, slicenum] = results[4]

        #   K1[:,:,slicenum],K2[:,:,slicenum],K3[:,:,slicenum],K4[:,:,slicenum],V[:,:,slicenum],alpha1Img[:,:,slicenum],alpha2Img[:,:,slicenum]=calculateK(sizeX, sizeY, alpha1, alpha2,index,M,W,TACs)
        K1_window = np.reshape(K1_window, [(__x2 - __x1), (__y2 - __y1), images.shape[3]])
        K2_window = np.reshape(K2_window, [(__x2 - __x1), (__y2 - __y1), images.shape[3]])
        K3_window = np.reshape(K3_window, [(__x2 - __x1), (__y2 - __y1), images.shape[3]])
        K4_window = np.reshape(K4_window, [(__x2 - __x1), (__y2 - __y1), images.shape[3]])

        V_window = np.reshape(V_window, [(__x2 - __x1), (__x2 - __x1), images.shape[3]])

        K1[__x1:__x2, __y1:__y2, :] = K1_window
        K2[__x1:__x2, __y1:__y2, :] = K2_window
        K3[__x1:__x2, __y1:__y2, :] = K3_window
        K4[__x1:__x2, __y1:__y2, :] = K4_window
        V[__x1:__x2, __y1:__y2, :] = V_window

        save_images(mypath, "r20", savePerfix, K1, K2, K3, K4, V)
#     save_images(mypath,"STC","TYPE2",K1,K2,K3 )