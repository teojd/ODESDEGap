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
from data_dists import *
import time



#Loss function
def pinn_loss(state):
  #Notation: - 1st character is the dimension of the problem
  #          - 2nd character is the dimension of the boundary
  #          - 3rd character indicates which boundary (left or right)
  #          - e.g. x_yl denotes the x (1) coordinates of points on the left (3) y boundary (2) 
  #            or y_tl is are the y coordinates corresponding to the initial condition (left t boundary)

  xdist = torch.distributions.Uniform(torch.tensor(state.xlim[0]).to(state.device),torch.tensor(state.xlim[1]).to(state.device))
  ydist = torch.distributions.Uniform(torch.tensor(state.ylim[0]).to(state.device),torch.tensor(state.ylim[1]).to(state.device))
  tdist = torch.distributions.Uniform(torch.tensor(state.tlim[0]).to(state.device),torch.tensor(state.tlim[1]).to(state.device))

  #Create a mesh of points inside the domain
  tt = tdist.sample([state.n_x,1]).requires_grad_()  #size [n_x,1]
  xx = xdist.sample([state.n_x,1]).requires_grad_()  #size [n_x,1]
  yy = ydist.sample([state.n_x,1]).requires_grad_()  #size [n_x,1]
  inputs  = torch.cat([tt,xx,yy],1)


  #Initial conditions
  x_tl = xdist.sample([state.n_0,1])  #size [n_0,1]
  y_tl = ydist.sample([state.n_0,1])  #size [n_0,1]
  t_tl = state.tlim[0]*torch.ones_like(x_tl,device=state.device, requires_grad=True)     #size [n_0,1]
  inputs0 = torch.cat([t_tl,x_tl,y_tl],1)         

  #Initial conditions
  x_tr = xdist.sample([state.n_0,1])  #size [n_0,1]
  y_tr = ydist.sample([state.n_0,1])  #size [n_0,1]
  t_tr = state.tlim[1]*torch.ones_like(x_tl,device=state.device, requires_grad=True)     #size [n_0,1]
  inputsT = torch.cat([t_tr,x_tr,y_tr],1)         



  # Boundary conditions
  # left y boundary
  x_yl = xdist.sample([state.n_b,1])  #size [n_b,1]
  y_yl = state.ylim[0]*torch.ones_like(x_yl,device=state.device, requires_grad=True)
  t_yl = tdist.sample([state.n_b,1])  #size [n_b,1]     
  X_yl = torch.cat([t_yl,x_yl,y_yl],1)         

  # right y boundary
  x_yr = xdist.sample([state.n_b,1])  #size [n_b,1]
  y_yr = state.ylim[1]*torch.ones_like(x_yr,device=state.device, requires_grad=True)
  t_yr = tdist.sample([state.n_b,1])  #size [n_b,1]    
  X_yr = torch.cat([t_yr,x_yr,y_yr],1)         

  # left x boundary
  y_xl = ydist.sample([state.n_b,1])  #size [n_b,1]
  x_xl = state.xlim[0]*torch.ones_like(y_xl,device=state.device, requires_grad=True)
  t_xl = tdist.sample([state.n_b,1])  #size [n_b,1]     
  X_xl = torch.cat([t_xl,x_xl,y_xl],1)         

  # right x boundary
  y_xr = ydist.sample([state.n_b,1])  #size [n_b,1]
  x_xr = state.xlim[1]*torch.ones_like(y_xr,device=state.device, requires_grad=True)
  t_xr = tdist.sample([state.n_b,1])  #size [n_b,1]     
  X_xr = torch.cat([t_xr,x_xr,y_xr],1)         

  #total boundary surface
  inputsb = torch.cat([X_xl,X_yl,X_xr,X_yr],0)    #size [4*n_b,3]


  #Evaluate network on these inputs
  if state.type=='FP':
      u  = state.net.forward(inputs)
      u0 = state.net.forward(inputs0)
      uT = state.net.forward(inputsT)
      ub = state.net.forward(inputsb)
  else:  
      u  = state.net.forward_log(inputs)
      u0 = state.net.forward_log(inputs0)
      uT = state.net.forward_log(inputsT)
      ub = state.net.forward_log(inputsb)


  #Calculate differential operator terms
  u_t  = grad(u,tt,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]   
  u_x  = grad(u,xx,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
  u_xx = grad(u_x,xx,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
  u_y  = grad(u,yy,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
  u_yy = grad(u_y,yy,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]

  if state.type=='FP':
      g0 = torch.exp(-((x_tl-2)**2+(y_tl+2)**2)) #+ torch.exp(-((x_tl+2)**2+(y_tl-2)**2))    
  else:
      g0 = -((x_tl-2)**2+(y_tl+2)**2) #+ torch.exp(-((x_tl+2)**2+(y_tl-2)**2))
  #g0 = -((x_tl-2)**2+(y_tl+2)**2)*3 #+ torch.exp(-((x_tl+2)**2+(y_tl-2)**2))    




  #Using computed terms, construct loss function and return this
  if state.type=='FP':
    res = torch.mean((u_t - 2*u - inputs[:,1].reshape([-1,1])*u_x - inputs[:,2].reshape([-1,1])*u_y - 0.5*(u_xx + u_yy))**2)
    bc = torch.mean(ub**2)
    ic  = torch.mean((u0 - g0)**2) #+ torch.mean((uT-gT)**2)
    loss= state.w_fp*res + state.w_ic*ic + state.w_b*bc
  else:
    res = torch.mean((u_t - 2 - inputs[:,1].reshape([-1,1])*u_x - inputs[:,2].reshape([-1,1])*u_y - 0.5*u_x**2 - 0.5*u_y**2 - 0.5*(u_xx + u_yy))**2)
    bc = torch.mean(torch.exp(ub)**2)
    ic  = torch.mean((u0 - g0)**2) #+ torch.mean((uT-gT)**2)
    loss= state.w_fp*res + state.w_ic*ic + state.w_b*bc
  
  return loss, res, ic, bc

def terminal_loss(state):
    xdist = torch.distributions.Uniform(torch.tensor(state.xlim[0]).to(state.device),torch.tensor(state.xlim[1]).to(state.device))
    ydist = torch.distributions.Uniform(torch.tensor(state.ylim[0]).to(state.device),torch.tensor(state.ylim[1]).to(state.device))
    x_tr = xdist.sample([state.n_0,1])  #size [n_0,1]
    y_tr = ydist.sample([state.n_0,1])  #size [n_0,1]
    t_tr = state.tlim[1]*torch.ones_like(x_tr,device=state.device, requires_grad=True)     #size [n_0,1]
    inputsT = torch.cat([t_tr,x_tr,y_tr],1)         

    gauss_dist = torch.distributions.MultivariateNormal(torch.tensor([0.,0.]).to(state.device), 0.5*torch.eye(2).to(state.device))

    #Evaluate network on these inputs
    if state.type=='FP':
        uT = state.net.forward(inputsT)
    else:  
        uT = state.net.forward_log(inputsT).flatten()
        pri_T = gauss_dist.log_prob(inputsT[:,1:]).flatten()
        loss = torch.mean(torch.abs(uT - pri_T))

        
    return loss



def plot_dens(state):
    state.net.to('cpu')
    runT = state.tlim_plt[1]
    n_x_plt = 40
    n_y_plt = 40
    n_t_plt = 301
    fps = 0.3*(n_t_plt-1)/runT


    xx = np.linspace(state.xlim_plt[0],state.xlim_plt[1],n_x_plt,dtype=np.float32).reshape([-1,1])
    yy = np.linspace(state.ylim_plt[0],state.ylim_plt[1],n_y_plt,dtype=np.float32).reshape([-1,1])
    tt = np.linspace(0,state.tlim_plt[1],n_t_plt,dtype=np.float32).reshape([-1,1])

    tt=tt.reshape(-1,)
    xx,yy  = np.meshgrid(xx,yy)
    z_t  = np.zeros(shape=[n_y_plt,n_x_plt,n_t_plt])



    #2d 
    for i in range(n_t_plt):
        plt_inputs = np.hstack([tt[i]*np.ones_like(xx.reshape([-1,1]),dtype=np.float32), xx.reshape([-1,1]), yy.reshape([-1,1])])
        plt_inputs = torch.tensor(plt_inputs).requires_grad_()
        z_t[:,:,i] = state.net.forward(torch.FloatTensor(plt_inputs)).detach().numpy().reshape([n_x_plt,n_y_plt])
        


    def update_plot(frame_number, z_t, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(xx, yy, z_t[:,:,frame_number], cmap=cm.coolwarm)
        title.set_text(f'{np.mean(z_t[:,:,frame_number])/np.mean(z_t)}')#,'Integral={}'.format(integrals[frame_number][0])])
#        title.set_text(['time={}'.format(round(100*tt[frame_number])/100)])#,'Integral={}'.format(integrals[frame_number][0])])
    #    title.set_text(['time={}'.format(np.mean(z_t[:,:,frame_number]))])#,'Integral={}'.format(integrals[frame_number][0])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90., azim=270)
    title = ax.set_title('t=')

    plot = [ax.plot_surface(xx, yy, z_t[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(np.min(z_t),np.max(z_t))
    ani = animation.FuncAnimation(fig, update_plot, n_t_plt, fargs=(z_t, plot), interval=200/fps)
    state.net.to(state.device)
    return ani

def plot_logdens(state):
    state.net.to('cpu')
    runT = state.tlim_plt[1]
    n_x_plt = 40
    n_y_plt = 40
    n_t_plt = 301
    fps = (n_t_plt-1)/runT


    xx = np.linspace(state.xlim_plt[0],state.xlim_plt[1],n_x_plt,dtype=np.float32).reshape([-1,1])
    yy = np.linspace(state.ylim_plt[0],state.ylim_plt[1],n_y_plt,dtype=np.float32).reshape([-1,1])
    tt = np.linspace(0,state.tlim_plt[1],n_t_plt,dtype=np.float32).reshape([-1,1])

    tt=tt.reshape(-1,)
    xx,yy  = np.meshgrid(xx,yy)
    z_t  = np.zeros(shape=[n_y_plt,n_x_plt,n_t_plt])



    #2d 
    for i in range(n_t_plt):
        plt_inputs = np.hstack([tt[i]*np.ones_like(xx.reshape([-1,1]),dtype=np.float32), xx.reshape([-1,1]), yy.reshape([-1,1])])
        plt_inputs = torch.tensor(plt_inputs).requires_grad_()
        z_t[:,:,i] = state.net.forward_log(torch.FloatTensor(plt_inputs)).detach().numpy().reshape([n_x_plt,n_y_plt])
        


    def update_plot(frame_number, z_t, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(xx, yy, z_t[:,:,frame_number], cmap=cm.coolwarm)
    #    title.set_text(['time={}'.format(round(100*tt[frame_number])/100)])#,'Integral={}'.format(integrals[frame_number][0])])
        title.set_text(['time={}'.format(np.mean(z_t[:,:,frame_number]))])#,'Integral={}'.format(integrals[frame_number][0])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90., azim=270)
    title = ax.set_title('t=')

    plot = [ax.plot_surface(xx, yy, z_t[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(np.min(z_t),np.max(z_t))
    ani = animation.FuncAnimation(fig, update_plot, n_t_plt, fargs=(z_t, plot), interval=200/fps)
    state.net.to(state.device)
    return ani


def plot_score(state,dim):
    state.net.to('cpu')
    runT = state.tlim_plt[1]
    n_x_plt = 40
    n_y_plt = 40
    n_t_plt = 301
    fps = (n_t_plt-1)/runT


    xx = np.linspace(state.xlim_plt[0],state.xlim_plt[1],n_x_plt,dtype=np.float32).reshape([-1,1])
    yy = np.linspace(state.ylim_plt[0],state.ylim_plt[1],n_y_plt,dtype=np.float32).reshape([-1,1])
    tt = np.linspace(0,state.tlim_plt[1],n_t_plt,dtype=np.float32).reshape([-1,1])

    tt=tt.reshape(-1,)
    xx,yy  = np.meshgrid(xx,yy)
    z_t  = np.zeros(shape=[n_y_plt,n_x_plt,n_t_plt])



    #2d 
    for i in range(n_t_plt):
        t_in = torch.tensor(tt[i]*np.ones_like(xx.reshape([-1,1]),dtype=np.float32))
        x_in = torch.tensor(np.hstack([xx.reshape([-1,1]), yy.reshape([-1,1])]))
        z_t[:,:,i] = state.net.score(x_in,t_in).detach().numpy().reshape([n_x_plt,n_y_plt,2])[:,:,dim]
        


    def update_plot(frame_number, z_t, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(xx, yy, z_t[:,:,frame_number], cmap=cm.coolwarm)
    #    title.set_text(['time={}'.format(round(100*tt[frame_number])/100)])#,'Integral={}'.format(integrals[frame_number][0])])
        title.set_text(['time={}'.format(np.mean(z_t[:,:,frame_number]))])#,'Integral={}'.format(integrals[frame_number][0])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90., azim=270)
    title = ax.set_title('t=')

    plot = [ax.plot_surface(xx, yy, z_t[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(np.min(z_t),np.max(z_t))
    ani = animation.FuncAnimation(fig, update_plot, n_t_plt, fargs=(z_t, plot), interval=200/fps)
    state.net.to(state.device)
    return ani

