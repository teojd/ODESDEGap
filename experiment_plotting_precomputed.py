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
#matplotlib qt
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

class state():
    def __init__(self,potential=True,dat_dist='checker'):
        self.type = 'logFP'
        self.device = device

        # How long to simulate for
        self.eps = 0.0001
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



#%% Plotting using pre computed weights. 


dists = ['blob','circles2','checker']
w_ps = [0.,0.1,1.0,10.0]


iteration = 100000
nfe = 800
n_samples = 3000000
grid_n = 64

fig,axs = plt.subplots(4,6)
fig.set_size_inches(10., 6.4)

wdists_ode_truth = np.zeros([4,3])
wdists_sde_truth = np.zeros([4,3])
wdists_sde_ode   = np.zeros([4,3])

i=0
grid = make_grid(grid_n)

print('Computing wasserstein distances of discretised dists. May take a few minutes...')
for dist in dists:
    j=0
    s = state(dat_dist=dist)

    weights_truth = np.load(f'precomputed_dists/weights_truth_{grid_n}_{n_samples}_{dist}.npy')

    for w_p in w_ps:
        s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device('cpu'))
        ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]

        weights_ode = np.load(f'precomputed_dists/weights_ode_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}.npy')
        weights_sde = np.load(f'precomputed_dists/weights_sde_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}.npy')

        axs[j,i].imshow(weights_sde.reshape([grid_n,grid_n]),extent=ext,origin='lower')
        #axs[j,i].set_title(f'{wdists[j,round(i/2)]:.2f}')
        axs[j,i].set_xticks([])
        axs[j,i].set_yticks([])

        axs[j,i+1].imshow(weights_ode.reshape([grid_n,grid_n]),extent=ext,origin='lower')
        axs[j,i+1].set_xticks([])
        axs[j,i+1].set_yticks([])

        print(f'{i}{j}')
        if i==0:
            axs[j,i].set_ylabel(r'$w_R$='+f'{w_p}')
            axs[j,i].set_ylabel(r'$w_R$='+f'{w_p}')

        if j==0:
            axs[j,i].set_title(r'$p_\theta^{SDE}(\cdot,0)$')
            axs[j,i+1].set_title(r'$p_\theta^{ODE}(\cdot,0)$')

        wdists_ode_truth[j,round(i/2)] = compute_wasserstein(weights_ode,weights_truth,grid)
        wdists_sde_truth[j,round(i/2)] = compute_wasserstein(weights_sde,weights_truth,grid)
        wdists_sde_ode[j,round(i/2)] = compute_wasserstein(weights_ode,weights_sde,grid)


        j=j+1
    i=i+2

print(wdists_ode_truth**2)
print(wdists_sde_truth**2)
print(wdists_sde_ode**2)
fig.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.05, 
                    hspace=0.05)
fig.savefig('sample_dists.png')



# %%
