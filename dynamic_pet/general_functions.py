from os import listdir
from os.path import join
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def get_images(mypath, prefix, big_slice, end_slice):
    print('get images ' + prefix)
    dynamic_images = [join(mypath, f) for f in listdir(mypath) if prefix in f]
    dynamic_images.sort()
    images = [nib.load(path).get_fdata() for path in dynamic_images[big_slice:end_slice]]
    images = np.asarray(images)
    if len(images.shape)==5:
        print(images.shape)
        images = nib.load(dynamic_images[0]).get_fdata()
        images=np.transpose(images, (3, 0, 1,2))
    images[np.isnan(images)]=0

    images[np.isinf(images)]=0



    return np.asarray(images)

def get_affine(mypath, prefix,word_in):
    print('get images ' + prefix)
    dynamic_images = [join(mypath, f) for f in listdir(mypath) if f.startswith(prefix) and word_in in f]
    affine = nib.load(dynamic_images[0]).affine
    return affine

def get_mask(mypath, file_name):
    return np.asarray(nib.load(join(mypath, file_name)).get_fdata())


def get_c_p(time,carotid_mask, big_slice, end_slice, mypath, prefix):
    images = get_images(mypath, prefix, big_slice, end_slice)
    i=int(images.shape[3]/4)
    c_p = np.asarray([np.mean(image[carotid_mask >0 ]) for image in images])
    c_p=c_p
    fig, ax = plt.subplots()

    ax.plot(time, c_p)

    ax.set_title('IDIF')

    ax.set_xlabel('time')

    ax.set_ylabel('con')

    fig.savefig(join(mypath,  "IDIF.png"))
    del images
    return np.asarray(c_p)

def get_tac(images,mask):
    c_p = np.asarray([np.mean(image[mask >= 1]) for image in images])
    return np.asarray(c_p)
def get_sigma(images,mask):
    
    sigma = np.asarray([np.std(image.flatten()) for image in images])
    return np.asarray(sigma)

def get_mean_from_mask(path,img_name,mask):
    
    img=nib.load(join(path,img_name)).get_fdata()
    temp=img[mask >= 1]
    c_p = np.mean(temp[temp >0])
    return c_p

def get_25_from_mask(path,img_name,mask):
    
    img=nib.load(join(path,img_name)).get_fdata()
    mask=get_mask(path, mask)
    temp=img[mask >= 1]
    c_p = temp[int(len(temp)*0.25)]
    return c_p

def get_75_from_mask(path,img_name,mask):
    
    img=nib.load(join(path,img_name)).get_fdata()
    mask=get_mask(path, mask)
    temp=img[mask >= 1]
    c_p = temp[int(len(temp)*0.75)]


    
    return c_p
def save_image(mypath, img, affine, file_name):
    array_img = nib.Nifti1Image(img.astype(np.float64), affine)
    nib.save(array_img, join(mypath, file_name))
