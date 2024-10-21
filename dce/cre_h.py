import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import torch
from monai.config import print_config
from monai.data import Dataset
from monai.losses import DiceCELoss
from monai.networks.layers import Norm
from monai.networks.nets import SegResNetVAE,SwinUNETR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from monai.networks.blocks import SubpixelUpsample
import numpy as np
from monai.data import PatchDataset, DataLoader, PatchIter

NUM_PATCH=384
max_iterations=100
a_min=0
NUM=1
lr=1e-3
num_samples=10
##model =UNet(
##        spatial_dims=2,
##        in_channels=1,
##        out_channels=1,
##        channels= (32,64 ),
##        strides=(2,),
##    )

##model = SwinUNETR((NUM_PATCH,NUM_PATCH),1,1,spatial_dims=2)
#model =SegResNetVAE((NUM_PATCH,NUM_PATCH),spatial_dims=2,norm=('GROUP', {'num_groups': 8}),out_channels=1)
model = SubpixelUpsample(spatial_dims=2, in_channels=1, out_channels=1,scale_factor=2 )


from monai.transforms import (

    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    SpatialCrop,
    Spacingd,
)
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
)

from reg import ants_apply_transform_for_masks_reverse,ants_apply_transform_for_masks
import os
from glob import glob


import numpy as np

from monai.data import PatchDataset, DataLoader, PatchIter
def get_inds(img):

    inds=[]
    if img.shape[0]==1:
        img=img[0]
    x,y,z=img.shape
    for i in range(NUM_PATCH,x+1,NUM_PATCH):
        for j in range(NUM_PATCH,y+1,NUM_PATCH):
            for k in range(z):
                inds.append([i,j,k])

    return inds


def train(global_step, train_loader, dice_val_best, global_step_best):
    #model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader)
    

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"], batch["label"])
        logit_map = model(x[:,:,:,:,0])
        loss = loss_function(logit_map, y[:,:,:,:,0])#+loss_function2(logit_map, y[:,:,:,:,0])
        loss.backward()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    return global_step, dice_val_best, global_step_best
subjs_folder = 'C:\\Users\\liron\\Downloads\\copy\\' #69

subject_folders = [folder for folder in os.listdir(subjs_folder) if os.path.isdir(os.path.join(subjs_folder, folder)) if 'sub' in folder ]
for subj_folder in subject_folders:

##    
##    model.load_state_dict(torch.load('C:\\Users\\liron\\Desktop\\manifest-gJIZVVFt6412408718812805737\\PROSTATE-DIAGNOSIS\\unetr.pth', map_location=torch.device('cpu')) )
##    model.eval()

    high_paths=[]
    low_paths=[]
    mypath = os.path.join(subjs_folder, subj_folder)
    dynPET_path = os.path.join(mypath, 'dynPET')
    dce_path = os.path.join(mypath, 'DCE')
    anatomy_path = os.path.join(mypath, 'T2')

    files_dynPET = glob(os.path.join(dce_path, 'blood.nii*'))
    files_dynPET2 = glob(os.path.join(dce_path, '*2deg*'))

    if not files_dynPET or not files_dynPET2:
        continue

    files_dce = glob(os.path.join(mypath, 'dtWarped.nii.gz'))

    ants_apply_transform_for_masks(subjs_folder, subj_folder, '20', files_dce[0].split('\\')[-1], 'dt', 'r', False)
    low_files=[os.path.join(mypath,file) for file in os.listdir(mypath) if 'dtWarped' in file and not file.startswith('r')]
    img=nib.load(low_files[0])
    array_img = nib.Nifti1Image(img.get_fdata(), img.affine)
    nib.save(array_img, os.path.join(subjs_folder,subj_folder,'DCE', 'T1_low.nii'))

    high_files=[os.path.join(anatomy_path,file) for file in os.listdir(anatomy_path) if 'nii' in file ]
    img=nib.load(high_files[0])
    I=img.get_fdata()[101:485,50:434,23:45]
    NUM=np.mean(I[I>0])+3*np.std(I[I>0])
#    print(NUM)

    array_img = nib.Nifti1Image(img.get_fdata()[101:485,50:434,23:45], img.affine)
    nib.save(array_img, os.path.join(subjs_folder,subj_folder,'DCE', 'T1_high.nii'))
    high_paths.append(os.path.join(subjs_folder,subj_folder,'DCE', 'T1_high.nii'))
    low_paths.append(os.path.join(subjs_folder,subj_folder,'DCE', 'T1_low.nii'))


    T2_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),

            ScaleIntensityRanged(
                keys=["image", "label"],
                a_min=a_min,
                a_max=NUM,
                b_min=0,
                b_max=1.0,
                clip=True,
            ),

             RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=(NUM_PATCH,NUM_PATCH,1), #64

                num_samples=num_samples,

            ),

            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.1
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1] ,
                prob=0.1,
            ),
     

        ]
    )

    train_transforms =  Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=NUM, b_min=0, b_max=1.0, clip=True)

            #CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    from monai.losses import SSIMLoss
    from monai.losses import PerceptualLoss

    loss_function= PerceptualLoss(2, network_type='alex')#,is_fake_3d=False)

    loss_function2=SSIMLoss(3)#,network_type='alex'
    #loss_function2= PerceptualLoss(2, network_type='radimagenet_resnet50' ).to(device)

    ##torch.backends.cudnn.benchmark = True




    ##


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)





        



    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(low_paths, high_paths)]




    train_ds =Dataset(
        data=train_files,
        transform=T2_transforms,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    global_step=0
    for i in range (max_iterations):
            #lr=scheduler(i, lr)
            global_step, dice_val_best, global_step_best = train(global_step, train_loader, 0, 0)
    ##
##for subj_folder in subject_folders:
##
##    low_files=[os.path.join(dce_path,file) for file in os.listdir(dce_path) if 'dyn' in file and  'nii.g' in file]
##    file1='C:\\Users\\liron\\Desktop\\manifest-gJIZVVFt6412408718812805737\\PROSTATE-DIAGNOSIS\\sr\\low\\rrPROSTATE-DIAGNOSIS_T1W_TSE_AX_CLEAR_20080708091130_801.nii'
    file2 = glob(os.path.join(mypath, 'dtInverseWarped.nii.gz'))[0]
##    mypath = os.path.join(subjs_folder, subj_folder)
##    dynPET_path = os.path.join(mypath, 'dynPET')
##    dce_path = os.path.join(mypath, 'DCE')
##    anatomy_path = os.path.join(mypath, 'T2')
##
##    files_dynPET = glob(os.path.join(dynPET_path, 'rpc*'))
##    files_dynPET2 = glob(os.path.join(dce_path, '*2deg*'))
####
##    if not files_dynPET or not files_dynPET2:
##        continue
    affine = nib.load(file2).affine  #.header.get_data_shape()
    label=torch.tensor(np.zeros((1,384,384,20)))


    


    files_dce = glob(os.path.join(dce_path, '*2deg*.nii*'))
    i=0



   # ants_apply_transform_for_masks_reverse(subjs_folder, subj_folder, '20', files_dce[0].split('\\')[-1], 'dt', 'r', False)
    low_files=[os.path.join(dce_path,file) for file in os.listdir(dce_path) if '2deg' in file and ( file.startswith('20') or file.startswith('d'))  ]

    data_dicts = [{"image": low_files[0]}]
    train_ds =Dataset(
        data=data_dicts,
        transform=train_transforms,
    )
    img = train_ds[0]["image"]
    indsf=get_inds(label)
    for  k in range(label.shape[3]):         
            x=img[None,:,:,:,k].to(device)
            label[:,:,:,k]=model(x)*NUM
    del img,train_ds
   # os.remove(os.path.join(subjs_folder,subj_folder,'DCE', 'r'+  files_dce[i].split('\\')[-1])) 
    array_img = nib.Nifti1Image(label.detach().cpu().numpy().astype(np.float32)[0], affine)
    nib.save(array_img, os.path.join(subjs_folder,subj_folder,'DCE', 'rr2flips'))
    del array_img
    del label

    
    files_dce = glob(os.path.join(dce_path, '*15deg*.nii*'))



   # ants_apply_transform_for_masks_reverse(subjs_folder, subj_folder, '20', files_dce[0].split('\\')[-1], 'dt', 'r', False)
    low_files=[os.path.join(dce_path,file) for file in os.listdir(dce_path) if '15deg' in file and ( file.startswith('20') or file.startswith('d'))]

    data_dicts = [{"image": low_files[0]}]
    train_ds =Dataset(
        data=data_dicts,
        transform=train_transforms,
    )
    img = train_ds[0]["image"]
    indsf=get_inds(img)
    label=torch.tensor(np.zeros((1,384,384,20)))
    for  k in range(label.shape[3]):
            
            x=img[None,:,:,:,k].to(device)
            label[:,:,:,k]=model(x)*NUM
    del img,train_ds
    #os.remove(os.path.join(subjs_folder,subj_folder,'DCE', 'r'+  files_dce[i].split('\\')[-1])) 
    array_img = nib.Nifti1Image(label.detach().cpu().numpy().astype(np.float32)[0], affine)
    nib.save(array_img, os.path.join(subjs_folder,subj_folder,'DCE', 'rr15flips'))

    files_dce = glob(os.path.join(dce_path, '*dyns*_000*.nii'))


    for i in range(35):

        #ants_apply_transform_for_masks_reverse(subjs_folder, subj_folder, '20', files_dce[i].split('\\')[-1], 'dt', 'r', False)
        low_files=[os.path.join(dce_path,file) for file in os.listdir(dce_path) if '_000' in file and 'nii' in file and ( file.startswith('20') or file.startswith('dy'))]

        data_dicts = [{"image": low_files[i]}]
        train_ds =Dataset(
            data=data_dicts,
            transform=train_transforms,
        )
        img = train_ds[0]["image"]
        
        for  k in range(label.shape[3]):
            
            x=img[None,:,:,:,k].to(device)
            label[:,:,:,k]=model(x)*NUM
        del img,train_ds
        #os.remove(os.path.join(subjs_folder,subj_folder,'DCE', 'r'+  files_dce[i].split('\\')[-1])) 
        array_img = nib.Nifti1Image(label.detach().cpu().numpy().astype(np.float32)[0], affine)
        nib.save(array_img, os.path.join(subjs_folder,subj_folder,'DCE', 'rr'+  files_dce[i].split('\\')[-1]))


    label=np.zeros((1,384,384,20,35))

    for i in range(35):
        label[:,:,:,:,i]=nib.load(os.path.join(subjs_folder,subj_folder,'DCE', 'rr'+  files_dce[i].split('\\')[-1])).get_fdata()
        os.remove(os.path.join(subjs_folder,subj_folder,'DCE', 'rr'+  files_dce[i].split('\\')[-1])) 

    array_img = nib.Nifti1Image(label[0], affine)
    nib.save(array_img, dce_path+'//high.nii.gz')

  
