import scipy
import skimage
from os.path import join
import nibabel as nib
import numpy as np
from dicoms.dynamic_pet import get_variables_from_dicom
from os import listdir


class HyprLr:
    """

    Dynamic PET Denoising with HYPR Processing
    https://jnm.snmjournals.org/content/jnumed/51/7/1147.full.pdf
    Improved kinetic analysis of dynamic PET data with optimized HYPR-LR
    """
    

    def __init__(self, mypath, nii_prefix, big_frame, end_frame, end_uptake, end_retention,images,affine):
        """

        :param mypath: path to nii files and dicom files. both of them need to be in the same folder!
        :param nii_prefix: prefix of nii files
        :param big_frame: number of  the frame from which starts the calculation
        :param end_frame: last frame that we insert to the calculation
        :param end_uptake: number of last frame in the uptake part
        :param end_retention: number of the last frame of the retention
        """
        self.mypath = mypath
        self.end_uptake = end_uptake
        self.end_retention = end_retention
        self.big_frame = big_frame
        self.end_frame = end_frame
        self.nii_prefix = nii_prefix
        self.images=images
        self.results =np.ones(( images.shape[1],images.shape[2],images.shape[3],images.shape[0]))
        self.affine=affine
    def _get_images_and_affine(self):
        """
         1. get images from nifti files- each nifti file neet to be 3D and should contain one frame
         2.convert them to numpy array
         3. get the affine (we need it to save the new images)
        :return: images,affine
        """

        print('set_images_and_affine')
        dynamic_imaging = [join(self.mypath, f) for f in listdir(self.mypath) if f.startswith(self.nii_prefix) and '00.nii' not in f]
        dynamic_imaging.sort()
        images = [nib.load(path).get_fdata() for path in dynamic_imaging[self.big_frame:self.end_frame]]
        images = np.asarray(images)
        images[images<0]=0
        images[np.isnan(images)]=0
        affine = nib.load(dynamic_imaging[23]).affine
        return images, affine

    def _get_f(self, s,size=3):
        """
        :param size: the size of one side of the symmetric  3d boxcar kernel
        set f- the low pass spatial filter function (sizexsizexsize boxcar smoothing filter)
        :return: none

        # skimage.filters.window('boxcar',[size,size,size])

s        """
        return skimage.filters.window('boxcar',size)

    def _get_i_c(self, images, slices_duration, i_c_shape):
        print('_get_i_c')

        i_c = np.zeros(i_c_shape)
        for image, slice_duration in zip(images, slices_duration):
            i_c = i_c + np.dot(slice_duration, image)
        return i_c

    def _get_slices_duration(self):
        """
        :return:slices_duration
        """
        print('get slices_duration')

        # data_from_dicom= get_variables_from_dicom(self.mypath, self.big_frame, self.end_frame)
        slicesDuration = [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.5, 0.5, 0.5, 0.5,
                          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.83333333, 0.83333333, 0.83333333, 0.83333333
            , 0.83333333, 0.83333333, 5., 4.]
        return slicesDuration  # np.array(data_from_dicom['sl3ices_duration'])

    def create_hypr_lr_images(self):
        """
        create_hypr_lr_image
        using scipy.ndimage.convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0)
        notice: F*I(x,y)=sum(sum(F(i,j)I(x-i,y-j))
        :return: none
        """
    
        slices_duration = self._get_slices_duration()
        print(slices_duration)

        # uptake
        self.create_images_for_one_part(self.images[0:self.end_uptake + 1,:,:,:], self.affine, slices_duration[0:self.end_uptake+ 1] ,
                                        0)
        # retention
        self.create_images_for_one_part(self.images[self.end_uptake + 1:self.end_retention + 1], self.affine,
                                        slices_duration[self.end_uptake + 1:self.end_retention + 1],self.end_uptake+ 1)
##        # equilibrium
        if self.end_retention<23:
               self.create_images_for_one_part(self.images[self.end_retention + 1:24,:,:,:], self.affine,
                                        slices_duration[self.end_retention + 1:24],self.end_retention+ 1)
        array_img = nib.Nifti1Image(self.results.astype(np.float64), self.affine)
        nib.save(array_img, join(self.mypath, 'motcorrW.nii.gz'))



    def create_images_for_one_part(self,images,affine, slices_duration, big_index):
        i_c = self._get_i_c(images, slices_duration, images[0].shape)
        print(images[0].shape)
        if 256 in images[0].shape:
             f = self._get_f(9/2.355,[6,6,9])#6 6 8
        else:
            f = self._get_f(9/2.355,[4,4,9])#4 4 8

##            
##            f = self._get_f([4,4,8])#7
##        else:
##            f = self._get_f([5,5,8])#

     
        denominator = scipy.ndimage.convolve(i_c, f)
        for num in range(len(images)):
            print('create_hypr_lr_images- image num: ' + str(big_index + num))
            numerator = scipy.ndimage.convolve(images[num,:,:,:], f)
            i_w = numerator / denominator
            i_h = i_c * i_w
            i_h[np.isnan(i_h)] = 0
            self.results[:,:,:,big_index + num]=i_h
