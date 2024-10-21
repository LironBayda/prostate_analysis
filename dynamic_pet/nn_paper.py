import os
import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math
import torch.optim as optim
import scipy
from sklearn.preprocessing import normalize
from glob import glob

from general_functions import *


#path= 'C:\\Users\\liron\\Downloads\\torn\\dp300_w_n.csv'

#path_n= 'C:\\Users\\liron\\Downloads\\torn\\dp300_w_n.csv'


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=500,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.002,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

 
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=20,
        spline_order=3,
        scale_noise=0.01,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.2,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )




class VAE(nn.Module):
    def __init__(self, input_dim = 120, latent_dim = 1 ):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.convLayer1 = nn.Conv1d(1,2, kernel_size=2)
        self.maxPool1d1 = nn.MaxPool1d(2, return_indices=True)


        self.convLayer2 = nn.Conv1d(2, 3, kernel_size=2)
        self.maxPool1d2 = nn.MaxPool1d(2, return_indices=True)
##
        self.convLayer3 = nn.Conv1d(3, 3, kernel_size=2)
        self.maxPool1d3 = nn.MaxPool1d(2, return_indices=True)


        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        
      
        
        self.fc_mu = nn.Linear(3, latent_dim, bias=True)
        self.fc_logvar =nn.Linear(3, latent_dim, bias=True)

        # Decoder layers
   
        self.fc_decode = nn.Linear( latent_dim,3, bias=True)
        self.unflatten = nn.Unflatten(1, (3, -1))

        self.maxUnPool1d3 = nn.MaxUnpool1d(2)
        self.convTransposeLayer3 = nn.ConvTranspose1d(3, 3, kernel_size=3)

        self.maxUnPool1d2 = nn.MaxUnpool1d(2)
        self.convTransposeLayer2 = nn.ConvTranspose1d(3, 2, kernel_size=2)

        self.maxUnPool1d1 = nn.MaxUnpool1d(2)
        self.convTransposeLayer1 = nn.ConvTranspose1d(2, 1, kernel_size=3)
        self.m = nn.LeakyReLU(0.5)
    def encode(self, x):
        #Encoder block 1
        x = self.m(self.convLayer1(x))

        x, indices1 = self.maxPool1d1(x)

##        # Encoder block 2
        x = self.m(self.convLayer2(x))

        x, indices2 = self.maxPool1d2(x)
##
        # Encoder block 3
        x = self.m(self.convLayer3(x))

        x, indices3 = self.maxPool1d3(x)

        # Flatten and calculate mu and logvar for latent space
        x = self.flatten(x)

      
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, (indices1,indices2,indices3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, indices):
        # Decode the latent vector
   

        z = self.fc_decode(z)
        z = self.unflatten(z)

        # Decoder block 1

        z = self.maxUnPool1d3(z, indices[2])


        z = self.m(self.convTransposeLayer3(z))
##
##        # Decoder block 2
##
        z = self.maxUnPool1d2(z, indices[1])

        z = self.m(self.convTransposeLayer2(z))

        # Decoder block 3
        z = self.maxUnPool1d1(z, indices[0])

        z = torch.sigmoid(self.convTransposeLayer1(z))

        return z

    
    def forward(self, x):
        mu, logvar,indices = self.encode(x) 
        z = self.reparameterize(mu, logvar) 
        return self.decode(z,indices), mu, logvar


# Loss Function
def loss_function(recon_x, x,mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



def get_tac_resample(dynamic_pet_path,mask_name,images,mid):
    mask = get_mask(dynamic_pet_path,mask_name )
    tac = get_tac(images, mask)
    if 'blood' in mask:
        tac=tac*1.63
    df= pd.Series(tac,index=mid)

    return df.resample('30s').bfill()[:17]
##
##
##        


##
##class AE(nn.Module):
##    def __init__(self, input_dim=1, hidden_dim=1,**keyword):
##        super(AE, self).__init__()
##        
##        # Encoder
##        self.encoder = nn.Sequential(
####            nn.Linear(36, 60
####                      ),
####            nn.Tanh(),
##            nn.Linear(60, 3),
##            nn.Tanh(),
##            nn.Linear(6, 5),
##            nn.Tanh(),
##            nn.Linear(5, 4),
##            nn.Tanh(),
##            nn.Linear(4, 3),
##            nn.Tanh(),
##            nn.Linear(3, 2),
##            nn.Tanh(),
##            nn.Linear(2, 1),
##            nn.Tanh(),
##
##
##            #nn.Dropout(0.8)
##        )
##        
##        # Decoder
##        self.decoder = nn.Sequential(
##
##            nn.Linear(1, 2),
##            nn.Tanh(),
##            nn.Linear(2, 3),
##            nn.Tanh(),
##            nn.Linear(3, 4),
##            nn.Tanh(),
##            nn.Linear(4, 5),
##            nn.Tanh(),
##            nn.Linear(5, 6),
##            nn.Tanh(),
##            nn.Linear(3, 6),
##            nn.Tanh(),
####            nn.Linear(60, 120),
####            nn.Tanh(),
##           # nn.Dropout(0.8)
##        )
##
##    def forward(self, x):
##        # Encode
##        encoded = self.encoder(x)
##        # Decode
##        decoded = self.decoder(encoded)
##        return decoded
##
##    def get_code(self, x):
##        return self.encoder(x)


def div(x,y):
    div=torch.autograd.grad(
                y, x,
                grad_outputs=torch.ones_like(y),
                retain_graph=True,
                create_graph=True
                )[0]

    return div

def get_sum_w(model):
    for n,p in enumerate(model.parameters()):
        sum_W=torch.tensor(0)
        sum_W=sum_W+torch.sqrt(torch.sum(torch.pow(p,2)))
    return sum_W
        





def circle_conv(tac_cancer_resample,blood_cancer_resample):
    tac_cancer_resample = scipy.fft.ifft(scipy.fft.fft(tac_cancer_resample)/scipy.fft.fft(blood_cancer_resample))
    tac_cancer_resample=np.real(tac_cancer_resample)
    tac_cancer_resample=np.append(tac_cancer_resample[15:],( tac_cancer_resample[:15]))
##    tac_cancer_resample[tac_cancer_resample<0]=0
    return tac_cancer_resample


import torch


class TACDataset:
    def __init__(
        self,
        path,
    ):
        """
        :param image_paths: list of paths to NIfTI images
        :param time_index: Pandas date_range or list representing the time points for TACs
        :param resize: tuple (512, 512) or None for resizing (if needed)
        :param augmentations: albumentations augmentations (if needed)
        :param channel_first: if True, convert TACs to channel-first format
        :param time_intervals: resampling interval for TAC (e.g., '20S' for 20 seconds)
        """
        self.path = path
        self.df = pd.read_csv(path).T
    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # Load the 3D NIfTI image (4D: x, y, z, time)
        patch = self.df.iloc[item].values.astype(np.float32)[:20]        # If needed, reshape the TAC to add a channel dimension for consistency
        patch=normalize(patch.reshape(1, -1))
        
        return {
            "tac": torch.tensor(patch, dtype=torch.float32)
        }



        

#if __name__ == '__main__':
mypath= 'E:\\copy'
image_paths = [
    glob(os.path.join(mypath, f, 'dynpet', '*000.nii.g*'))[0]  # Grab the first match from glob
    for f in os.listdir(mypath) if f.startswith('sub') and glob(os.path.join(mypath, f, 'dynPET/rpc*'))
]

epochs=10000001
num_point=61
model = VAE(input_dim=61, latent_dim=1)
time_index=['00:00:10','00:00:20','00:00:30','00:00:40','00:00:50','00:01:00'
                        ,'00:01:30','00:02:00','00:02:30','00:03:00','00:03:30','00:04:00',
                        '00:04:30','00:05:00','00:05:30','00:06:00',
                        '00:06:50','00:07:40','00:08:30','00:09:20','00:10:10',
                        '00:11:00','00:16:00','00:20:00']
time_index=pd.to_datetime(time_index)
optimizer = optim.SGD(model.parameters(),lr=0.001)#,nesterov=True, momentum=0.9,weight_decay=0.0001)


# Create the dataset instance


# Create the DataLoader

# Create the DataLoader

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

##
##model.load_state_dict(torch.load( os.path.join("D://model_paper(10).pth"),map_location=torch.device('cpu')), strict=False)        


    
### Training loop
##for epoch in range(epochs):
##    # Iterate over batches of data from the DataLoader
##    for batch_data  in data_loader:
##        # Extract the TAC batch and move it to the device
##        batch_tacs = batch_data['tac'].to(device)
##        if torch.sum(batch_tacs)<=0:
##            continue
##
##        # Forward pass: Compute model output (reconstructed TAC, mu, logvar)
##        reconstructed, mu, logvar = model(batch_tacs)
##
##        # Compute the loss (reconstruction loss + KL divergence)
##        train_loss = loss_function(reconstructed, batch_tacs, mu, logvar)
##        
##        # Zero the gradients before performing backpropagation
##        optimizer.zero_grad()
##
##        # Backward pass: Compute gradients of the loss w.r.t. model parameters
##        train_loss.backward()
##
##        # Update model parameters based on the current gradients
##        optimizer.step()
##    print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss.item():.4f}")
##    # Print the loss every 100 epochs
##    if epoch % 10 == 0 :
##    ##        
##    ##
##    ##
##    ##        
data={}
data_cancer,data_notcancer={},{}

mypath= 'E:\\copy'
paths = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.startswith('sub')] 
for path in paths:
    mid=time_index
    
    dynamic_pet_path = os.path.join(path, 'dynPET')

    files_dynPET = glob(os.path.join(dynamic_pet_path, 'pvc_pc*'))

##            if not files_dynPET:
##                continue
    images_noise = get_images(dynamic_pet_path, '000.nii.g', 0, 24)
##    images = get_images(dynamic_pet_path, 'motcorrW.nii.g', __big_slice, __end_slice)

    
    
    ##            else:
    for mask in ['rnotcancer_dixon.nii','rcancer_dixon.nii']:
##       
        tac_cancer_resample=get_tac_resample(dynamic_pet_path,mask,images_noise,time_index)
##
##        blood_cancer_resample=get_tac_resample(dynamic_pet_path,'pvc_blood_mask.nii',images_noise,mid)
##
##
##
##        tac_cancer_resample=circle_conv(tac_cancer_resample.values,blood_cancer_resample.values)
##
        tac_cancer_resample=normalize(tac_cancer_resample.values.reshape(1, -1))[0]
##
##
        batch_features=torch.reshape(torch.tensor(tac_cancer_resample,dtype=torch.float32),(1,1,-1))
##        o,_,_= model(batch_features)
##
##
##        plt.clf()
####
##        plt.plot(range(0,len(batch_features[0][0])*30,30),o.detach().numpy()[0][0])
##        plt.plot(range(0,len(batch_features[0][0])*30,30),batch_features[0][0],'o')
####
##        plt.savefig(dynamic_pet_path+'/atoencoder_'+mask[:-4])
##
        if 'not' in mask:
##
##
##            x_notcancer,_,_=model.encode(batch_features)
##            x_notcancer=x_notcancer.detach().numpy()
            data_notcancer[path.split('\\')[-1]]=list(batch_features.detach().numpy()[0][0])
        else:
##
##            
##            x_cancer,_,_=model.encode(batch_features)
##            x_cancer=x_cancer.detach().numpy()
            data_cancer[path.split('\\')[-1]]=list(batch_features.detach().numpy()[0][0])
##
##









##    data[path.split('\\')[-1]] = {
##                       'x1_pet_cancer': x_cancer[0][0],
##                     
####                              'x2_pet_cancer': x_cancer[0][1],
####                              'x3_pet_cancer': x_cancer[0][2],
####
####                              'x4_pet_cancer': x_cancer[0][3],
####                              'x5_pet_cancer': x_cancer[0][4],
####                                'x6_pet_cancer': x_cancer[0][5],
####                               
####                              'x7_pet_cancer': x_cancer[0][6],
####                              'x8_pet_cancer': x_cancer[0][7],
####
####                              'x9_pet_cancer': x_cancer[0][8],
####                              'x10_pet_cancer': x_cancer[0][9],
##
##                   #   'mse_cancer':mse_cancer,
##     
##                 
##
##
##                      
##                      
##                      'x1_pet_notcancer': x_notcancer[0][0],
####                              'x2_pet_notcancer': x_notcancer[0][1],
####                              'x3_pet_notcancer': x_notcancer[0][2],
####                              'x4_pet_notcancer': x_notcancer[0][3],
####                              'x5_pet_notcancer': x_notcancer[0][4],
####                               'x6_pet_notcancer': x_notcancer[0][5],
####                            
####                              'x7_pet_notcancer': x_notcancer[0][6],
####                              'x8_pet_notcancer': x_notcancer[0][7],
####
####                              'x9_pet_notcancer': x_notcancer[0][8],
####                              'x10_pet_notcancer': x_notcancer[0][9],
####        ##                                      'mse_notcancer':mse_notcancer
####
##                       }
##                         
##
##
##
##
##
##df = pd.DataFrame(data)
##df=df.T
##df.to_csv('D:\\atoencoder120.csv')

df = pd.DataFrame(data_notcancer)
df=df.T
df.to_csv('D:\\data_notcancer17_000.csv')

df = pd.DataFrame(data_cancer)
df=df.T
df.to_csv('D:\\data_cancer17_000.csv')

##torch.save(model.state_dict(), os.path.join("D://best_metric_model_PET_ae2.pth"))


