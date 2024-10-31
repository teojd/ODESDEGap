import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from models import Net
from data_dists import *
import ot
from scipy.spatial.distance import cdist
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def transition(x0,t,state):
    mean_t = x0*torch.exp(state.OU_rate*t)
#    variance_t = (1/(-2*OU_rate))*(1-torch.exp(2*OU_rate*t))
    variance_t = 0.5*(1-torch.exp(2*state.OU_rate*t))
    variance_t = variance_t.unsqueeze(-1)*torch.eye(2).to(device).unsqueeze(0)
    transition_dist = torch.distributions.MultivariateNormal(mean_t, variance_t)
    xt = transition_dist.sample()
    xt = xt.requires_grad_()
    kernel_xt = transition_dist.log_prob(xt)
    kernel_score = grad(kernel_xt,xt,grad_outputs=torch.ones_like(kernel_xt),retain_graph=True,create_graph=True)[0]  
    return xt, kernel_score

def denoising_loss(state):
    x0 = sample_dat_dist(state.n_denoise,state).to(device)
    t  = torch.tensor(np.random.beta(1,2,[state.n_denoise,1]).astype(np.float32)).to(device)*state.tlim[1]+ state.tlim[0]
    #t  = torch.tensor(np.random.uniform(state.tlim[0],state.tlim[1],[state.n_denoise,1]).astype(np.float32)).to(device)
    xt, kernel_score_xt = transition(x0,t,state)
    score_xt = state.net.score(xt,t)
    #loss = torch.mean(torch.exp(-t)*(score_xt-kernel_score_xt)**2)
    loss = torch.mean((score_xt-kernel_score_xt)**2)
    return loss

def simulate_rsde(xT,nfe,state):
    x = xT.reshape([-1,2])
    t  = state.tlim_plt[1]
    dt = torch.tensor(-state.tlim_plt[1]/nfe,device=device)
    ones = torch.ones([x.shape[0],1]).to(device)
    breakpoint = round(x.shape[0]/2)
    for i in range(nfe-10):
        if i%100==0:
            dW = torch.randn([100,x.shape[0],x.shape[1]],device=device)
        dx1 = (-x[:breakpoint] - state.net.score(x[:breakpoint],t*ones[:breakpoint]))*dt + torch.sqrt(-dt)*dW[i%100][:breakpoint]
        dx2 = (-x[breakpoint:] - state.net.score(x[breakpoint:],t*ones[breakpoint:]))*dt + torch.sqrt(-dt)*dW[i%100][breakpoint:]
        dx = torch.concat([dx1,dx2],axis=0)
        x = x + dx
        t = t + dt
        x = x.detach()
    dt = dt/10
    dW = torch.randn([100,x.shape[0],x.shape[1]],device=device)
    for i in range(99):
        dx1 = (-x[:breakpoint] - state.net.score(x[:breakpoint],t*ones[:breakpoint]))*dt + torch.sqrt(-dt)*dW[i][:breakpoint]
        dx2 = (-x[breakpoint:] - state.net.score(x[breakpoint:],t*ones[breakpoint:]))*dt + torch.sqrt(-dt)*dW[i][breakpoint:]
        dx = torch.concat([dx1,dx2],axis=0)
        x = x + dx
        t = t + dt
        x = x.detach()
    dx1 = (-x[:breakpoint] - state.net.score(x[:breakpoint],t*ones[:breakpoint]))*dt
    dx2 = (-x[breakpoint:] - state.net.score(x[breakpoint:],t*ones[breakpoint:]))*dt
    dx = torch.concat([dx1,dx2],axis=0)
    x = x + dx
    return x

def simulate_rode(xT,nfe,state):
    x = xT.reshape([-1,2])
    t  = state.tlim_plt[1]
    dt = -state.tlim_plt[1]/nfe
    ones = torch.ones([x.shape[0],1]).to(device)
    breakpoint = round(x.shape[0]/2)
    for i in range(nfe-10):
        dx1 = (-x[:breakpoint] - 0.5*state.net.score(x[:breakpoint],t*ones[:breakpoint]))*dt
        dx2 = (-x[breakpoint:] - 0.5*state.net.score(x[breakpoint:],t*ones[breakpoint:]))*dt
        dx = torch.concat([dx1,dx2],axis=0)
        x = x + dx
        t = t + dt
        x = x.detach()
    dt = dt/10
    for i in range(100):
        dx1 = (-x[:breakpoint] - 0.5*state.net.score(x[:breakpoint],t*ones[:breakpoint]))*dt
        dx2 = (-x[breakpoint:] - 0.5*state.net.score(x[breakpoint:],t*ones[breakpoint:]))*dt
        dx = torch.concat([dx1,dx2],axis=0)
        x = x + dx
        t = t + dt
        x = x.detach()       
        x = x.detach()
    return x

def validation_plot(n_samples,state,iteration=0,grid_n = 30):
    nfe = 400
    truncation_b = 4
    ext = [state.xlim_plt[0], state.xlim_plt[1], state.ylim_plt[0], state.ylim_plt[1]]
    alph = 200/np.min([n_samples,20000])

    print(1)
    xT = torch.FloatTensor(np.random.normal(0,np.sqrt(0.5),[n_samples,2]).astype(np.float32)).to(device)
    sde_samples = simulate_rsde(xT,nfe,state).cpu().detach().numpy()
    np.random.shuffle(sde_samples)
    plt.subplot(221)
    plt.scatter(sde_samples[:20000,0],sde_samples[:20000,1],alpha=alph,s=20)
    plt.xlim(state.xlim_plt)
    plt.ylim(state.ylim_plt)
    plt.title('sde samples')

    print(2)
    xT = torch.FloatTensor(np.random.normal(0,np.sqrt(0.5),[n_samples,2]).astype(np.float32)).to(device)
    ode_samples = simulate_rode(xT,nfe,state).cpu().detach().numpy()
    np.random.shuffle(ode_samples)
    plt.subplot(222)
    plt.scatter(ode_samples[:20000,0],ode_samples[:20000,1],alpha=alph,s=20)
    plt.xlim(state.xlim_plt)
    plt.ylim(state.ylim_plt)
    plt.title('ode samples')

    print(3)
    plt.subplot(223)
    plt.scatter(ode_samples[:20000,0],ode_samples[:20000,1],alpha=alph,s=20)
    plt.scatter(sde_samples[:20000,0],sde_samples[:20000,1],alpha=alph,s=20)
    plt.xlim(state.xlim_plt)
    plt.ylim(state.ylim_plt)
    plt.title('sde and ode samples')

    print(4)
    plt.subplot(224)
    x0 = sample_dat_dist(n_samples,state).detach().numpy()
    np.random.shuffle(x0)
    plt.scatter(x0[:20000,0],x0[:20000,1],alpha=alph,s=20)
    plt.xlim(state.xlim_plt)
    plt.ylim(state.ylim_plt)
    plt.title('ground truth samples')

    ode_samples = ode_samples[np.abs(ode_samples[:,0])<truncation_b]
    ode_samples = ode_samples[np.abs(ode_samples[:,1])<truncation_b]

    sde_samples = sde_samples[np.abs(sde_samples[:,0])<truncation_b]
    sde_samples = sde_samples[np.abs(sde_samples[:,1])<truncation_b]

    n_truncated = np.min([ode_samples.shape[0],sde_samples.shape[0]])
    ode_samples = ode_samples[:n_truncated]
    sde_samples = sde_samples[:n_truncated]
    np.random.shuffle(x0)
    x0 = x0[:n_truncated]

    # discretise dists into bins to compute wasserstein with large sample sizes
    weights_ode, grid = discretise_dist(ode_samples,grid_n=grid_n,truncation=truncation_b)
    weights_sde, _    = discretise_dist(sde_samples,grid_n=grid_n,truncation=truncation_b)
    weights_truth, _  = discretise_dist(x0,grid_n=grid_n,truncation=truncation_b)
    wdist = compute_wasserstein(weights_ode,weights_sde,grid)


    fig = plt.gcf()
    if iteration!=0: 
        state.writer.add_figure('Samples',fig, iteration)
        state.writer.add_scalars(f'Wasserstein distance. {state.n_denoise} points', {'train':wdist}, iteration)
    else:
        wdist_otruth = compute_wasserstein(weights_truth,weights_ode,grid)
        wdist_truth = compute_wasserstein(weights_truth,weights_sde,grid)
        print(f'ODE-SDE wasserstein distance is {wdist}')
        print(f'ODE-Truth wasserstein distance is {wdist_otruth}')
        print(f'SDE-Truth wasserstein distance is {wdist_truth}')
    state.net.to(state.device)

    #plt.show(block=suppress_plot)
    plt.show()

    ulim = np.max(np.vstack([weights_sde,weights_truth]))
    ulim_ode = np.max(weights_ode)

    plt.figure()
    fig = plt.gcf()

    plt.subplot(221)
    weights_sde = weights_sde.reshape([grid_n,grid_n])
    plt.imshow(weights_sde, norm=colors.SymLogNorm(linthresh=0.00005, linscale=0.00005,vmin=0.0, vmax=ulim),extent=ext,origin='lower')
    plt.colorbar()

    plt.subplot(222)
    weights_ode = weights_ode.reshape([grid_n,grid_n])
    plt.imshow(weights_ode, norm=colors.SymLogNorm(linthresh=0.00005, linscale=0.00005,vmin=0.0, vmax=ulim),extent=ext,origin='lower')
    plt.colorbar()

    plt.subplot(224)
    weights_truth = weights_truth.reshape([grid_n,grid_n])
    plt.imshow(weights_truth, norm=colors.SymLogNorm(linthresh=0.00005, linscale=0.00005,vmin=0.0, vmax=ulim),extent=ext,origin='lower')
    plt.colorbar()

    plt.subplot(223)
    ode_sde_error = weights_ode - weights_sde
    plt.imshow(ode_sde_error, norm=colors.SymLogNorm(linthresh=0.00005, linscale=0.00005,vmin=-ulim, vmax=ulim),extent=ext,origin='lower')
    plt.colorbar()
    #plt.show(block=suppress_plot)
    plt.show()

    if iteration!=0: 
        state.writer.add_figure('densities',fig, iteration)


def discretise_dist(samples, grid_n = 30, truncation = 4.):
    n_samples = samples.shape[0]
    n_chunks = 100
    chunk_size = round(n_samples/n_chunks)+1
    grid1, grid2 = np.meshgrid(np.linspace(-truncation,truncation,grid_n),np.linspace(-truncation,truncation,grid_n))
    grid = np.hstack([grid1.reshape([-1,1]),grid2.reshape([-1,1])])
    weights = np.zeros(grid.shape[0])

    for i in range(n_chunks):
        samples_chunk = samples[i*chunk_size:(i+1)*chunk_size]
        M = cdist(samples_chunk,grid)**2
        nearest_n = np.argmin(M,1)
        inds,counts = np.unique(nearest_n,return_counts=True)   
        weights[inds] = weights[inds] + counts
        print(f'1{i}')

    weights = weights/np.sum(weights)
    return weights, grid

def make_grid(grid_n = 30, truncation = 4.):
    grid1, grid2 = np.meshgrid(np.linspace(-truncation,truncation,grid_n),np.linspace(-truncation,truncation,grid_n))
    grid = np.hstack([grid1.reshape([-1,1]),grid2.reshape([-1,1])])
    return grid

def compute_wasserstein(weights1, weights2, grid,type=2):
    M1 = cdist(grid,grid)
    if type==1:
        wdist = ot.emd2(weights1,weights2,M1, numItermax=10000000,numThreads='max')
    elif type==2:
        M2 = M1**2
        wdist = np.sqrt(ot.emd2(weights1,weights2,M2, numItermax=10000000,numThreads='max'))
    return wdist