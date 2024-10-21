import os
from os import listdir
from os.path import join

import numpy as np
import nibabel as nib


class BilateralFilter:

    def __init__(self):
        self.gaussian_p_window = None
        self.k_side = None
        self.window_index = None
        self.__anat_folder, self.__anat_prefix = 'SUB_A_G_839', 'pProstate_MRAC_PET_diff_W.nii.gz'

        self.__dyn_pet_folder, self.__dyn_pet_prefix = 'dynPet', '000.nii.gz'

        self.__win_x1, self.__win_y1, self.__win_z1 = 0, 0, 0

        self.__win_x2, self.__win_y2, self.__win_z2 = 172, 172 , 127
        self.kernel_size =5
        self.k_side=2
        self.sigma_p = 9/2.33
        self.sigma_i = 800
        self.__output_name = 'bf'

    def save_images(self, path, template, img, affine, i):
        template[self.__win_x1:self.__win_x2, self.__win_y1:self.__win_y2,
        self.__win_z1:self.__win_z2] = img
        array_img = nib.Nifti1Image(template, affine)
        if i < 10:
            nib.save(array_img, os.path.join(path, self.__dyn_pet_folder, self.__output_name + '0' + str(i) + '.nii'))
        else:
            nib.save(array_img, os.path.join(path, self.__dyn_pet_folder, self.__output_name + str(i) + '.nii'))

    def get_images(self, mypath, prefix):
        print(mypath)
        dynamic_imaging = [join(mypath, f) for f in listdir(mypath) if prefix in f]
        images = nib.load(dynamic_imaging[0]).get_fdata() 
        images = np.asarray(images)
        images[images<0]=0
        print(images.shape)
        print(dynamic_imaging)

        return images, nib.load(dynamic_imaging[0]).affine

    def gaussian_p(self, m, n):
        return np.exp(-np.power((m - n), 2) / (2 * np.power(self.sigma_p, 2)))

    def gaussian_I(self, m, n):

        return np.exp(-np.power(np.sqrt(np.power(m - n, 2)), 2) / (2 * np.power(self.sigma_i, 2)))

    def get_results(self, window_i):
        gaussian_i_window = self.gaussian_I(window_i[self.k_side, self.k_side],
                                            window_i[self.window_index[:, 0], self.window_index[:, 1]])
        n = np.sum(self.gaussian_p_window * gaussian_i_window *
                   window_i[self.window_index[:, 0], self.window_index[:, 1]])
        d = np.sum(self.gaussian_p_window * gaussian_i_window)
        return n / d

    def numerator(self, window_i):

        return np.sum(self.gaussian_p_window *
                      self.gaussian_I(window_i[self.k_side, self.k_side],
                                      window_i[self.window_index[:, 0], self.window_index[:, 1]]) *
                      window_i[self.window_index[:, 0], self.window_index[:, 1]])

    def denominator(self, window_i):

        return np.sum(self.gaussian_p_window *
                      self.gaussian_I(window_i[self.k_side, self.k_side],
                                      window_i[self.window_index[:, 0], self.window_index[:, 1]]))

    def bilateral_filter(self, image_i):
        x, y, z = image_i.shape
        self.k_side = int((self.kernel_size - 1) / 2)
        th = 0
        results = np.zeros([x, y, z])
        inds = np.indices(image_i[self.k_side:x-self.k_side,self.k_side:y-self.k_side,self.k_side:z-self.k_side].shape).reshape(3, -1)+self.k_side

        for  ind in inds.T:
            if image_i[ind[0],ind[1],ind[2]]>0:
                self.sigma_i = (3.5 * image_i[ind[0],ind[1],ind[2]])
                results[ind[0],ind[1],ind[2]]=self.get_results\
            (image_i[ind[0]-self.k_side:ind[0]+self.k_side+1, ind[1]-self.k_side:ind[1]+self.k_side+1, ind[2]])


        return results



    def denoise(self, my_path):
        #        image_p, affine = self.get_images(my_path + '/' + self.__anat_folder, self.__anat_prefix)
        image_i, affine = self.get_images(my_path + '/' + self.__dyn_pet_folder, self.__dyn_pet_prefix)
        print(image_i.shape)
        x_ind=2*int(image_i.shape[0]/5)
        y_ind=2*int(image_i.shape[1]/5)
        self.__win_x1=x_ind
        self.__win_y1=y_ind
        self.__win_x2=image_i.shape[0]-x_ind
        self.__win_y2=image_i.shape[1]-y_ind
        self.window_index = np.asarray([[ii, jj, kk]
                                        for ii in range(0, self.kernel_size)
                                        for jj in range(0, self.kernel_size)
                                        for kk in range(0, self.kernel_size) if
                                        ii != self.k_side or jj != self.k_side or kk != self.k_side])
        self.gaussian_p_window = self.gaussian_p( self.k_side, 4.17 * self.window_index[:, 0]) * \
                                 self.gaussian_p(4.17 *self.k_side, 4.17 * self.window_index[:, 1]) * \
                                 self.gaussian_p(2 * self.k_side, 2 * self.window_index[:, 2])

        for i in range(24):

            image_i[ self.__win_x1:self.__win_x2, self.__win_y1:self.__win_y2,
                                            self.__win_z1:self.__win_z2,i] = self.bilateral_filter(image_i[ self.__win_x1:self.__win_x2, self.__win_y1:self.__win_y2,
                                            self.__win_z1:self.__win_z2,i])
        array_img = nib.Nifti1Image(image_i, affine)
        nib.save(array_img, os.path.join(my_path, self.__dyn_pet_folder, self.__output_name + '.nii'))
