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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


gauss_dist = torch.distributions.MultivariateNormal(torch.tensor([2.,-2.]), 0.2*torch.eye(2))
gauss_dist2 = torch.distributions.MultivariateNormal(torch.tensor([-2.,2.]), 0.2*torch.eye(2))
gauss_dist3 = torch.distributions.MultivariateNormal(torch.tensor([2.,2.]), 0.2*torch.eye(2))
gauss_dist4 = torch.distributions.MultivariateNormal(torch.tensor([-2.,-2.]), 0.2*torch.eye(2))
gauss_dist5 = torch.distributions.MultivariateNormal(torch.tensor([0.,0.]), 0.2*torch.eye(2))
gauss_dist6 = torch.distributions.MultivariateNormal(torch.tensor([4.,0.]), 0.2*torch.eye(2))
gauss_dist7 = torch.distributions.MultivariateNormal(torch.tensor([-4.,0.]), 0.2*torch.eye(2))
gauss_dist8 = torch.distributions.MultivariateNormal(torch.tensor([0.,4.]), 0.2*torch.eye(2))
gauss_dist9 = torch.distributions.MultivariateNormal(torch.tensor([0.,-4.]), 0.2*torch.eye(2))

gauss_dists = [gauss_dist,gauss_dist2,gauss_dist3,gauss_dist4,gauss_dist5,gauss_dist6,gauss_dist7,gauss_dist8,gauss_dist9]

def gauss_blob(n,n_blob = 9):
    samples = []
    for i in range(n_blob):
        samples.append(gauss_dists[i].sample([int(n/n_blob)+1]))
    samples = torch.cat(samples)*0.65
    randperm = torch.randperm(samples.shape[0])
    samples = samples[randperm]
    return samples[:n]

def checker(n):
    xx = np.random.uniform(-0.5,-0.25,[int(n/8),1])
    yy = np.random.uniform(-0.5,-0.25,[int(n/8),1])
    xy0 = np.hstack([xx,yy])
    xy = xy0
    shifts = [[0.5,0],[0.25,0.25],[0.75,0.25],[0,0.5],[0.5,0.5],[0.25,0.75],[0.75,0.75]]
    for shift in shifts:
       xy = np.vstack([xy,xy0+shift])
    return torch.tensor(xy.astype(np.float32))*6
        

def circles(n):
   r = np.random.normal(1.,0.1,[n,1]).astype(np.float32)
   theta = np.random.uniform(0.,2*np.pi,[n,1]).astype(np.float32)
   xx = r*np.cos(theta).reshape([-1,1])
   yy = r*np.sin(theta).reshape([-1,1])
   xy = np.hstack([xx,yy])
   return torch.tensor(xy)


def circles2(n):
   n = round(n/2)
   r1 = np.random.normal(0.65,0.1,[n,1]).astype(np.float32)
   r2 = np.random.normal(1.4,0.1,[n,1]).astype(np.float32)
   theta = np.random.uniform(0.,2*np.pi,[n,1]).astype(np.float32)
   xx1 = r1*np.cos(theta).reshape([-1,1])
   yy1 = r1*np.sin(theta).reshape([-1,1])
   xx2 = r2*np.cos(theta).reshape([-1,1])
   yy2 = r2*np.sin(theta).reshape([-1,1])
   xy1 = np.hstack([xx1,yy1])
   xy2 = np.hstack([xx2,yy2])
   xy = np.vstack([xy1,xy2])
   return torch.tensor(xy)



def sample_dat_dist(n,state):
   if state.dat_dist=='checker':
       return checker(n)
   if state.dat_dist=='blob':
       return gauss_blob(n)
   if state.dat_dist=='circles':
       return 2*circles(n)
   if state.dat_dist=='circles2':
       return 2*circles2(n)