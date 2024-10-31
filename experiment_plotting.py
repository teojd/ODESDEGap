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




def generate_sde_samples(n_samples,nfe,state):
    xT = torch.FloatTensor(np.random.normal(0,np.sqrt(0.5),[n_samples,2]).astype(np.float32)).to(device)
    sde_samples = simulate_rsde(xT,nfe,state).cpu().detach().numpy()
    np.random.shuffle(sde_samples)
    sde_samples = sde_samples[np.abs(sde_samples[:,0])<state.xlim_plt[1]]
    sde_samples = sde_samples[np.abs(sde_samples[:,1])<state.xlim_plt[1]]
    return sde_samples

def generate_ode_samples(n_samples,nfe,state):
    xT = torch.FloatTensor(np.random.normal(0,np.sqrt(0.5),[n_samples,2]).astype(np.float32)).to(device)
    ode_samples = simulate_rode(xT,nfe,state).cpu().detach().numpy()
    np.random.shuffle(ode_samples)
    ode_samples = ode_samples[np.abs(ode_samples[:,0])<state.xlim_plt[1]]
    ode_samples = ode_samples[np.abs(ode_samples[:,1])<state.xlim_plt[1]]
    return ode_samples
    


#%% Sample from all distributions and make plots (gpu recommended, experminent_plotting_precomputed.py)
# Loops through all pre trained models, samples from their ode and sde distribution,
# computes the weights of their discretised distribution, plots these and computes W2 dists

dists = ['blob','circles2','checker']
w_ps = [0.0,0.1,1.0,10.0]


iteration = 100000
nfe = 800
n_samples = 50000 # 3000000 # uncomment for full sample size used in paper 
grid_n = 64

n_loops = round(n_samples/25000) # round(n_samples/300000) # performed in batches due to memory constraints. Uncomment for full sample size used in paper 

wdists_ode_truth = np.zeros([4,3])
wdists_sde_truth = np.zeros([4,3])
wdists_sde_ode   = np.zeros([4,3])

fig,axs = plt.subplots(4,6)
fig.set_size_inches(12.5, 8)

wdists = np.zeros([4,3])

i=0
for dist in dists:
    j=0
    s = state(dat_dist=dist)

    true_samples = sample_dat_dist(n_samples,s).detach().numpy()
    weights_truth, grid  = discretise_dist(true_samples,grid_n=grid_n,truncation=s.xlim[1])
    np.save(f'weights_truth_{grid_n}_{n_samples}_{dist}',weights_truth)
    np.save(f'grid_{grid_n}',grid)

    weights_truth = np.load(f'weights_truth_{grid_n}_{n_samples}_{dist}.npy')
    grid = np.load(f'grid_{grid_n}.npy')


    print(0)
    for w_p in w_ps:
        sde_samples_list = []
        ode_samples_list = []
        s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device(device))
        s.tlim_plt[1] = 10.
        ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]

        for k in range(n_loops):
            sde_samples_list.append(generate_sde_samples(round(n_samples/n_loops),nfe,s))
            ode_samples_list.append(generate_ode_samples(round(n_samples/n_loops),nfe,s))
            print(f'0{k}')

        sde_samples = np.vstack(sde_samples_list)
        ode_samples = np.vstack(ode_samples_list)

        weights_ode, _ = discretise_dist(ode_samples ,grid_n=grid_n,truncation=s.xlim[1])
        weights_sde, _    = discretise_dist(sde_samples ,grid_n=grid_n,truncation=s.xlim[1])

        np.save(f'weights_ode_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}',weights_ode)
        np.save(f'weights_sde_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}',weights_sde)

        wdists[j,round(i/2)] = compute_wasserstein(weights_ode,weights_sde,grid)

        axs[j,i].imshow(weights_sde.reshape([grid_n,grid_n]),extent=ext,origin='lower')
        #axs[j,i].set_title(f'{wdists[j,round(i/2)]:.2f}')
        axs[j,i].set_xticks([])
        axs[j,i].set_yticks([])

        axs[j,i+1].imshow(weights_ode.reshape([grid_n,grid_n]),extent=ext,origin='lower')
        axs[j,i+1].set_xticks([])
        axs[j,i+1].set_yticks([])

        print(f'{i}{j}')
        if i==0:
            axs[j,i].set_ylabel(f'{w_p}')
            axs[j,i].set_ylabel(f'{w_p}')

        if j==0:
            axs[j,i].set_title(f'SDE')
            axs[j,i+1].set_title(f'ODE')

        wdists_ode_truth[j,round(i/2)] = compute_wasserstein(weights_ode,weights_truth,grid)
        wdists_sde_truth[j,round(i/2)] = compute_wasserstein(weights_sde,weights_truth,grid)
        wdists_sde_ode[j,round(i/2)] = compute_wasserstein(weights_ode,weights_sde,grid)


        j=j+1
        del s.net
    i=i+2

print(wdists)
fig.tight_layout()
fig.savefig('sample_dists.png')
np.savetxt(f'distances_{n_samples}_{grid_n}_{nfe}',np.array(wdists),delimiter=',')







#%% residual vs w2 distance

distances = np.genfromtxt('distances_64_3000000_100000_800',delimiter=',')
#distances = wdists_sde_ode
distances = distances**2
resids    = np.genfromtxt('FP_resid_100000',delimiter=',')

plt.scatter(distances[:,0],resids[:,0],c='b',label='Mixture',marker="+")
plt.scatter(distances[:,1],resids[:,1],c='r',label='Circles',marker="*")
plt.scatter(distances[:,2],resids[:,2],c='g',label='Checkerboard',marker="^")
plt.yscale('log')
plt.xscale('log')

plt.legend()

plt.xlabel(r'$W^2_2(p_\theta^{SDE}(\cdot,0),p_\theta^{ODE}(\cdot,0))$')
plt.ylabel(r'$\tilde{R}(\theta,\phi_\theta,0)$')

x = np.array([236,0.07,0.007,0.0009])
y = np.array([0.1638,0.09387,0.06304,0.05695])

plt.savefig('resids_x_dists.png')

print(wdists_ode_truth**2)
print(wdists_sde_truth**2)
print(wdists_sde_ode**2)

#%% FPR DSM scatter
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt


dists = ['blob','circles2','checker']
w_ps = [0.,0.1,1.0,10.0]

means = np.zeros([4,3])

i=0
for dist in dists:
    j=0
    for w_p in w_ps:
        a = pd.read_csv(f'tb_data/run-{dist}_w{w_p}_Denoising Loss Joint 200000_train-tag-Denoising Loss Joint 200000.csv')['Value']
        means[j,i] = np.mean(a[800:])
        j+=1
        print(j)
    i+=1


#plt.plot(means)

def norm(x):
    return (x - np.min(x))/(np.max(x)-np.min(x)) -0.5

plt.scatter(norm(loss_denoise00001[:,0]),resids[:,0],c='b',label='Mixture',marker="+")
plt.scatter(norm(loss_denoise00001[:,1]),resids[:,1],c='r',label='Circles',marker="*")
plt.scatter(norm(loss_denoise00001[:,2]),resids[:,2],c='g',label='Checkerboard',marker="^")

plt.yscale('log')
#plt.xscale('log')

plt.legend()


plt.xlabel(r'$\bar{\mathcal{L}}_{DSM}(\theta,\nabla \phi_\theta,\lambda)$')
plt.ylabel(r'$\tilde{R}(\theta,\phi_\theta,0)$')

plt.savefig('resids_x_DSM.png')


#%% Plotting using pre computed weights


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
loss_denoise00001 = np.zeros([4,3])

lis = []

i=0
for dist in dists:
    j=0
    s = state(dat_dist=dist)

    weights_truth = np.load(f'weights_truth_{grid_n}_{n_samples}_{dist}.npy')
    grid = np.load(f'grid_{grid_n}.npy')

    for w_p in w_ps:
        s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device('cpu'))
        for k in range(100):
            lis.append(denoising_loss(s).detach().numpy())
        loss_denoise00001[j,i] = np.mean(lis)
        print(w_p)
        j=j+1
    i=i+1

        ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]

        weights_ode = np.load(f'weights_ode_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}.npy')
        weights_sde = np.load(f'weights_sde_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}.npy')

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

        #wdists_ode_truth[j,round(i/2)] = compute_wasserstein(weights_ode,weights_truth,grid)
        #wdists_sde_truth[j,round(i/2)] = compute_wasserstein(weights_sde,weights_truth,grid)
        #wdists_sde_ode[j,round(i/2)] = compute_wasserstein(weights_ode,weights_sde,grid)


        j=j+1
    i=i+2

print(wdists_ode_truth)
print(wdists_sde_truth)
print(wdists_sde_ode)
fig.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.05, 
                    hspace=0.05)
fig.savefig('sample_dists.png')



#%% ODE Truth distance


#%% Sliced version

dists = ['blob','circles2','checker']
w_ps = [0.0,0.1,1.0,10.0]


iteration = 100000
nfe = 800
n_samples = 5000000
grid_n = 80

wdists_ode_truth = np.zeros([4,3])
wdists_sde_truth = np.zeros([4,3])
wdists_sde_ode   = np.zeros([4,3])

a, b = np.ones((n_samples,)) / n_samples, np.ones((n_samples,)) / n_samples
#a = a.to(device)
#b = b.to(device)
i=0
for dist in dists:
    j=0
    s = state(dat_dist=dist)

    true_samples = sample_dat_dist(n_samples,s).detach().numpy()

    for w_p in w_ps:
        s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}')
        s.tlim_plt[1] = 10.
        ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]

        sde_samples  = generate_sde_samples(n_samples,nfe,s)
        ode_samples  = generate_ode_samples(n_samples,nfe,s)

        print(f'{i}{j}')
        wdists_ode_truth[j,i] = ot.sliced_wasserstein_distance(ode_samples, true_samples, a, b, 20) 
        wdists_sde_truth[j,i] = ot.sliced_wasserstein_distance(sde_samples, true_samples, a, b, 20) 
        wdists_sde_ode[j,i] = ot.sliced_wasserstein_distance(sde_samples, ode_samples, a, b, 20) 

        j=j+1
    i=i+1

print(wdists_ode_truth)
print(wdists_sde_truth)
print(wdists_sde_ode)
print('')
print(wdists_ode_truth**2)
print(wdists_sde_truth**2)
print(wdists_sde_ode**2)


#%% Separate ode and sde sample plot
'''

dists = ['blob','circles2','checker']
w_ps = [0.0,0.1,1.0,10.0]


iteration = 100000
nfe = 200
n_samples = 100
grid_n = 10

sde_fig,sde_axs = plt.subplots(4,3)
sde_fig.set_size_inches(8.5, 10)

ode_fig,ode_axs = plt.subplots(4,3)
ode_fig.set_size_inches(8.5, 10)

wdists = np.zeros([4,3])

i=0
for dist in dists:
    j=0
    s = state(dat_dist=dist)
    true_samples = sample_dat_dist(n_samples,s).detach().numpy()
    weights_truth, grid  = discretise_dist(true_samples,grid_n=grid_n,truncation=s.xlim[1])
    np.save(f'weights_truth_{grid_n}_{n_samples}',weights_truth)
    np.save(f'grid_{grid_n}',grid)

    for w_p in w_ps:
        s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device('cpu'))
        s.tlim_plt[1] = 10.
        ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]



        sde_samples  = generate_sde_samples(n_samples,nfe,s)
        ode_samples  = generate_ode_samples(n_samples,nfe,s)

        weights_ode, _ = discretise_dist(ode_samples ,grid_n=grid_n,truncation=s.xlim[1])
        weights_sde, _    = discretise_dist(sde_samples ,grid_n=grid_n,truncation=s.xlim[1])

        np.save(f'weights_ode_{grid_n}_{n_samples}_{iteration}_{nfe}',weights_truth)


        wdists[j,i] = compute_wasserstein(weights_ode,weights_sde,grid)

        sde_axs[j,i].imshow(weights_sde.reshape([grid_n,grid_n]),extent=ext,origin='lower')
        sde_axs[j,i].set_title(f'{wdists[j,i]:.2f}')
        sde_axs[j,i].set_xticks([])
        sde_axs[j,i].set_yticks([])

        ode_axs[j,i].imshow(weights_ode.reshape([grid_n,grid_n]),extent=ext,origin='lower')
        ode_axs[j,i].set_title(f'{wdists[j,i]:.2f}')
        ode_axs[j,i].set_xticks([])
        ode_axs[j,i].set_yticks([])

        if i==0:
            print(1)
            sde_axs[j,i].set_ylabel(f'{w_p}')
            sde_axs[j,i].set_ylabel(f'{w_p}')
            
        j=j+1
    i=i+1

sde_fig.savefig('sde_samples.png')
ode_fig.savefig('ode_samples.png')



#%% Individual plot



dist = 'blob'
w_p = 10.0
iteration = 100000
n_val_pts = 1000
grid_n = 64


s = state(w_p,dat_dist=dist)
s.net = torch.load(f'checkpoints/{dist}/w{w_p}/{dist}_net__FP_weight_{w_p}__iteration_{iteration}',map_location=torch.device('cpu'))
s.tlim_plt[1] = 5.
ext = [s.xlim_plt[0], s.xlim_plt[1], s.ylim_plt[0], s.ylim_plt[1]]

##%matplotlib qt
#%matplotlib tk

nfe = 400
n_samples = 10000


sde_samples  = generate_sde_samples(n_samples,nfe,s)
ode_samples  = generate_ode_samples(n_samples,nfe,s)
true_samples = sample_dat_dist(n_samples,s).detach().numpy()

weights_truth, grid  = discretise_dist(true_samples,grid_n=grid_n,truncation=s.xlim[1])

weights_ode, _ = discretise_dist(ode_samples ,grid_n=grid_n,truncation=s.xlim[1])
weights_sde, _    = discretise_dist(sde_samples ,grid_n=grid_n,truncation=s.xlim[1])
wdist = compute_wasserstein(weights_ode,weights_sde,grid)

plt.subplot(121)
plt.imshow(weights_sde.reshape([grid_n,grid_n]),extent=ext,origin='lower')
plt.title(wdist)

plt.subplot(122)
plt.imshow(weights_ode.reshape([grid_n,grid_n]),extent=ext,origin='lower')
plt.title(wdist)


#%%


#Sample validation plot
validation_plot(n_val_pts,s,grid_n=grid_n)

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

'''
w_p = 0.1
np.save(f'weights_ode_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}',2*(weights_ode_01-weights_ode_0/2))
np.save(f'weights_sde_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}',2*(weights_sde_01-weights_sde_0/2))


w_p = 1.
np.save(f'weights_ode_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}',3*(weights_ode_1-weights_ode_01*2/3))
np.save(f'weights_sde_{grid_n}_{n_samples}_{iteration}_{nfe}_{dist}_{w_p}',3*(weights_sde_1-weights_sde_01*2/3))
'''