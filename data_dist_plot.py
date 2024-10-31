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
writer = SummaryWriter()
##%matplotlib qt
#%matplotlib tk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

class state():
    def __init__(self,potential=True,dist='blob'):
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
        self.dat_dist = dist

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


def discretise_dist(samples, grid_n = 30, truncation = 4.):
    grid1, grid2 = np.meshgrid(np.linspace(-truncation,truncation,grid_n),np.linspace(-truncation,truncation,grid_n))
    grid = np.hstack([grid1.reshape([-1,1]),grid2.reshape([-1,1])])

    M = cdist(samples,grid)**2
    nearest_n = np.argmin(M,1)
    inds,counts = np.unique(nearest_n,return_counts=True)   
    weights = np.zeros(grid.shape[0])
    weights[inds] = counts
    weights = weights/np.sum(weights)
    return weights, grid




grid_n = 64
fig,axs = plt.subplots(1,3)

dists = ['blob','circles2','checker']
dist_tit = ['Gaussian mixture', 'Circles', 'Checkerboard']
ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]

n_samples = 3000000 if device == 'cuda' else 100000

for i in range(3):
    ax = axs.ravel()[i]
    s = state(dist = dists[i])
    x0 = sample_dat_dist(n_samples,s)
    w0,g0 = discretise_dist(x0,grid_n)
    ax.imshow(w0.reshape([grid_n,grid_n]),extent=ext)
    ax.set_title(dist_tit[i])
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig('target_dists.png') 


# %%
