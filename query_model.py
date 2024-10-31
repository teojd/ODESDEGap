#%% Script to generate samples from the query model, plot distributions from ODE
# and SDE. Compute the Wasserstein distance between the two, and plot the outputs.
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

######################################################################
# SET OPTIONS HERE:
# choose data distribution/model 
model_name = 'circles2' #'blob','circles2' or 'checker'

# choose regularisation weight
w_p = 0.0 # 0.0, 0.1, 1.0 or 10.0

# choose number of samples to generate for the query model
n_samples = 10000
######################################################################




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
        self.dat_dist = model_name

        # network loaded into state variable too
        self.net = Net(potential=True).to(device)

        # add writer for images 
        self.writer = writer


s = state()




s.dat_dist = model_name
dist = s.dat_dist
w_p = w_p

iteration = 100000
n_val_pts = n_samples

s = state()
s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device(s.device))

validation_plot(n_val_pts,s,grid_n = 64)


#%% Can be run in interactive mode to visualise the evolution of the density as an animation
'''
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
'''


# %%