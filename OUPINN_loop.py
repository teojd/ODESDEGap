#%%
import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from models import Net
from diffusion_utils import *
from pinn_utils import *
import ot
import os 
import io 

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

class state():
    def __init__(self,w_p=-1,potential=True,dat_dist='checker'):
        self.type = 'logFP'
        self.device = device

        # How long to simulate for
        self.eps = 0.00001
        self.tlim = [self.eps,10.]

        # PINN boundaries
        self.xlim = [-3.5,3.5]
        self.ylim = [-3.5,3.5]

        # Number of PINN collocation points
        self.n_x = 50000
        self.n_b = 10000
        self.n_0 = 10000

        # Weightings for PINN terms
        self.w_fp = 1.
        self.w_ic = 1.
        self.w_b  = 1.
        

        # Diffusion iteration
        self.iter_diff = 0

        # Diffusion drift and diffusion constants
        self.OU_rate = torch.tensor(-1.).to(device)
        self.g = torch.tensor(1.).to(device)
        self.D = torch.tensor(0.5).to(device)
 
        # limits for validation plots
        self.tlim_plt = [self.eps,10.]
        self.xlim_plt = [-3.5,3.5]
        self.ylim_plt = [-3.5,3.5]

        # losses and percentile
        self.losses = []
        self.losses_90per = 1e5
        self.n_denoise = 200000

        # data distrubtion
        #self.dat_dist = 'circles2'
        #self.dat_dist = 'checker'
        #self.dat_dist = 'blob'
        self.dat_dist = dat_dist

        # network loaded into state variable too
        self.net = Net(potential=True,n_nodes=80).to(device)

        # add writer for images 
        if w_p == -1:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir=f'./summaries/{self.dat_dist}/w{w_p}')





#%% Joint training

# learning rate:iterations
lr_dict = {1e-3:25000,
           1e-4:65000,
           1e-5:10000
        }

dists = ['checker','circles2','blob']

### new inside ###
for w_p in [0.0,0.1,1.,10.]:
    for dist in dists:
        i=0
        s = state(w_p,dat_dist=dist)
        os.makedirs(f'./checkpoints/{s.dat_dist}/w{w_p}', exist_ok=True)
        if device.type == 'cpu':
            s.n_denoise = int(s.n_denoise/10)
            s.n_x = int(s.n_x/10)
            s.n_0 = int(s.n_0/10)
            s.n_b = int(s.n_b/10)

        ### this ###

        losses = []

        s.w_b = 0.  #boundary enforcement weight
        s.w_ic = 0.  #initial condition enforcement weight

        for lr in lr_dict:
            opt = torch.optim.Adam(s.net.parameters(), lr=lr)
            s.lr = lr
        
            for i in range(i+1,i+1+lr_dict[lr]):
                opt.zero_grad()
                loss_denoise = denoising_loss(s)
                loss = loss_denoise 
                if w_p > 0.:
                    loss_fp = pinn_loss(s)
                    loss = loss + w_p*loss_fp[0]
                loss.backward()
                opt.step()
                if i%10==0:
                    loss_pinn = pinn_loss(s)
                    print( (f'Iteration: {i}, '
                            f'Loss: {loss.cpu().detach():.2f}, '
                            f'Diff: {loss_denoise.cpu().item():.2f}, '
                            f'PINN: {loss_pinn[0].cpu().item():.2f}, '
                            f'Res: {loss_pinn[1].cpu().item():.2f}, '
                            f'IC: {loss_pinn[2].cpu().item():.2f}, '
                            f'BC: {loss_pinn[3].cpu().item():.2f}') )

                if i>0 and i%10000==0:
                    validation_plot(1750000,s,i,grid_n = 64)  
                elif lr == 1e-5 and i%2500==0:
                    validation_plot(1750000,s,i,grid_n = 64)  
                if i%10==0:
                    s.writer.add_scalars(f'PINN Loss Joint {s.n_denoise}', {'train':loss_pinn[1].cpu().detach().numpy()}, i)
                    s.writer.add_scalars(f'Denoising Loss Joint {s.n_denoise}', {'train':loss_denoise.cpu().detach().numpy()}, i)
                if i%1000==0 and i>0:
                    torch.save(s.net, f'./checkpoints/{s.dat_dist}/w{w_p}/{s.dat_dist}_net__FP_weight_{w_p}__iteration_{i}')


            del opt
        del s






#%% Diffusion only training

'''
lr = 0.0001
opt = torch.optim.Adam(s.net.parameters(), lr=lr)
s.lr = lr

s.iter_diff = i

for i in range(s.iter_diff,50000):
    opt.zero_grad()
    loss = denoising_loss(s)  
    loss.backward()
    s.losses.append(loss.cpu().detach().numpy())
    opt.step()

    # Make validation plot every 1000 iterations
    if i%500==499:
        losses_90per = np.percentile(np.array(s.losses[-2000:]),90)
        s.net.to('cpu')   
        validation_plot(2000,s,i)  
        s.net.to(device)   

    # Add scalar summary every 10 iterations
    if i%10==0:
      print('Iteration: %d, Loss: %.6f' % (i,loss.item()))
  #    if loss.item() < losses_90per:
      s.writer.add_scalars('Loss'+str(s.n_denoise)+' '+str(lr), {'train':loss.cpu().detach().numpy()}, i)
    s.iter_diff = i

#torch.save(s.net, 'network1_diffonly')


#%% PINN only training

lr = 0.001
opt = torch.optim.Adam(s.net.parameters(), lr=lr)
s.lr = lr

for i in range(i,10000):
    opt.zero_grad()
    loss = pinn_loss(s)    
    loss[0].backward()
    opt.step()
    if i%10==0:
      print('Iteration: %d, Loss: %.6f, Res: %.6f, IC: %.6f, BC: %.20f' % 
            (i,loss[0].item(),loss[1].item(),loss[2].item(),loss[3].item()))
      s.writer.add_scalars('PINN Loss'+str(lr), {'train':loss[0].cpu().detach().numpy()}, i)

opt = torch.optim.Adam(s.net.parameters(), lr=0.0001)

for i in range(i,10000):
    opt.zero_grad()
    loss = pinn_loss(s)
    loss[0].backward()
    opt.step()
    if i%10==0:
      print('Iteration: %d, Loss: %.6f, Res: %.6f, IC: %.6f, BC: %.10f' % 
            (i,loss[0].item(),loss[1].item(),loss[2].item(),loss[3].item()))
      s.writer.add_scalars('PINN Loss'+str(lr), {'train':loss[0].cpu().detach().numpy()}, i)





#%%
#s.net = torch.load('network_2',map_location=torch.device('cpu'))

##%matplotlib qt
#%matplotlib tk


#Sample validation plot
for j in range(10):
    n_val_pts = 10000
    validation_plot(n_val_pts,s)

s.tlim_plt = [s.eps,2.]

#Density plot
a = plot_dens(s)

#Logdensity plot
a = plot_logdens(s)

#score_x plot
a = plot_score(s,0)

#score_y plot
a = plot_score(s,1)


# %%


# torch.save(net, "network")

x0 = sample_dat_dist(s.n_denoise, s)
t  = torch.tensor(np.random.uniform(s.tlim[0],s.tlim[1],[s.n_denoise,1]))
t = torch.ones_like(t)*0.00001
xt, kernel_score_xt = transition(x0,t,s)
#plt.scatter(xt[:,0].detach().numpy(),xt[:,1].detach().numpy())

plt.quiver(xt[:,0].detach().numpy(),xt[:,1].detach().numpy(),kernel_score_xt[:,0].detach().numpy(),kernel_score_xt[:,1].detach().numpy())
plt.show()
xt = xt.float()
t = t.float()

plt.figure()
score_xt = s.net.score(xt,t)
plt.quiver(xt[:,0].detach().numpy(),xt[:,1].detach().numpy(),
           score_xt[:,0].detach().numpy(),score_xt[:,1].detach().numpy())
plt.show()
'''

