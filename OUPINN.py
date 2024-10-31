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

import io 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=f'./summaries')
##%matplotlib qt
#%matplotlib tk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

class state():
    def __init__(self,potential=True):
        self.type = 'logFP'
        self.device = device

        # How long to simulate for
        self.eps = 0.00001
        self.tlim = [self.eps,10.]
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
        self.n_denoise = 19980

        # data distrubtion
        self.dat_dist = 'circles2'

        if self.dat_dist=='blob':
            self.xlim_plt = [-5.5,5.5]
            self.ylim_plt = [-5.5,5.5]
            self.xlim = [-5.5,5.5]
            self.ylim = [-5.5,5.5]

        # network loaded into state variable too
        self.net = Net(potential=True).to(device)

        # add writer for images 
        self.writer = writer


s = state()


if device.type == 'cpu':
    s.n_denoise = int(s.n_denoise/10)
    s.n_x = int(s.n_x/10)
    s.n_0 = int(s.n_0/10)
    s.n_b = int(s.n_b/10)

i=0


#%% Joint training
lr = 0.001
opt = torch.optim.Adam(s.net.parameters(), lr=lr)
s.lr = lr
losses = []

s.w_b = 1.
s.w_ic = 0.

w_p = 0.
#s.net = torch.load('network0_joint0.001')

for i in range(i,1000000):
    opt.zero_grad()
    loss_denoise = denoising_loss(s)
    #term_loss = terminal_loss(s)
    loss = loss_denoise 
    if w_p > 0.:
        loss_pinn = pinn_loss(s)
        loss = loss + w_p*loss_pinn[0]# + term_loss
    loss.backward()
    opt.step()
    if i%10==0:
      if w_p == 0.:
          loss_pinn = pinn_loss(s) #eval every 10th iteration just for tracking
      print( (f'Iteration: {i}, '
              f'Loss: {loss.cpu().detach():.2f}, '
              f'Diff: {loss_denoise.cpu().item():.2f}, '
              f'PINN: {loss_pinn[0].cpu().item():.2f}, '
              f'Res: {loss_pinn[1].cpu().item():.2f}, '
              f'IC: {loss_pinn[2].cpu().item():.2f}, '
              f'BC: {loss_pinn[3].cpu().item():.2f}') )

    if i>0 and i%500==0:
        #losses_90per = np.percentile(np.array(losses[-2000:]),90)   
        validation_plot(5000,s,i,grid_n=50)
    if i%10==0:
      writer.add_scalars('PINN Loss Joint'+str(s.n_denoise)+' '+str(lr), {'train':loss_pinn[1].cpu().detach().numpy()}, i)
      writer.add_scalars('Denoising Loss Joint'+str(s.n_denoise)+' '+str(lr), {'train':loss_denoise.cpu().detach().numpy()}, i)
    if i%1000==0 and i>0:
        torch.save(s.net, f'./checkpoints/{s.dat_dist}/w{w_p}/{s.dat_dist}_net__FP_weight_{w_p}__iteration_{i}')
      




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
      writer.add_scalars('Loss'+str(s.n_denoise)+' '+str(lr), {'train':loss.cpu().detach().numpy()}, i)
    s.iter_diff = i

#torch.save(s.net, 'network1_diffonly')
'''

#%% PINN only training
'''
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
      writer.add_scalars('PINN Loss'+str(lr), {'train':loss[0].cpu().detach().numpy()}, i)

opt = torch.optim.Adam(s.net.parameters(), lr=0.0001)

for i in range(i,10000):
    opt.zero_grad()
    loss = pinn_loss(s)
    loss[0].backward()
    opt.step()
    if i%10==0:
      print('Iteration: %d, Loss: %.6f, Res: %.6f, IC: %.6f, BC: %.10f' % 
            (i,loss[0].item(),loss[1].item(),loss[2].item(),loss[3].item()))
      writer.add_scalars('PINN Loss'+str(lr), {'train':loss[0].cpu().detach().numpy()}, i)
'''




#%%
#s.net = torch.load('network_2',map_location=torch.device('cpu'))

##%matplotlib qt
#%matplotlib tk


#Sample validation plot
n_val_pts = 300000
validation_plot(n_val_pts,s,grid_n = 100)

for j in range(10):
    for n_val_pts in [100000,200000,300000]:
        for grid_n in [50,60,80,100]:
            t0 = time.time()
            #n_val_pts = 200000
            validation_plot(n_val_pts,s,grid_n = grid_n)
            t1 = time.time()
            print(f'npts {n_val_pts}, grid {grid_n}, time {t1-t0}')

s.tlim_plt = [s.eps,2.]

#%%

dist = 'blob'
w_p = 0.1

iteration = 100000

s = state()
s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device('cpu'))
s.tlim_plt = [0.,5.]




#Density plot
a = plot_dens(s)

#Logdensity plot
#a = plot_logdens(s)

#score_x plot
#a = plot_score(s,0)

#score_y plot
#a = plot_score(s,1)


# %%


# torch.save(net, "network")
s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=device)

x0 = sample_dat_dist(s.n_denoise, s).to(device)
t  = torch.tensor(np.random.uniform(s.tlim[0],s.tlim[1],[s.n_denoise,1])).to(device)
t = torch.ones_like(t)*0.00001
xt, kernel_score_xt = transition(x0,t,s)
#plt.scatter(xt[:,0].detach().numpy(),xt[:,1].detach().numpy())

plt.quiver(xt[:,0].to('cpu').detach().numpy(),xt[:,1].to('cpu').detach().numpy(),kernel_score_xt[:,0].to('cpu').detach().numpy(),kernel_score_xt[:,1].to('cpu').detach().numpy())
plt.show()
xt = xt.float()
t = t.float()

plt.figure()
score_xt = s.net.score(xt,t)
plt.quiver(xt[:,0].to('cpu').detach().numpy(),xt[:,1].to('cpu').detach().numpy(),
           score_xt[:,0].to('cpu').detach().numpy(),score_xt[:,1].to('cpu').detach().numpy())
plt.show()



# %%
