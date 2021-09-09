# -*- coding: utf-8 -*-
"""
This code provides necessary functions for the analysis of network connectivity and other properties
Created on Thu Feb 27 12:54:04 2020

@author: songt
"""

from __future__ import division

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import random

from numpy import linspace
import statsmodels.tsa.api as smt
from scipy import optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

#local the connectome data of macaque or marmoset network
def load_data(datafile):
    
    plt.close('all')
    
    with open(datafile,'rb') as f:
        p = pickle.load(f, encoding='latin1')   
        
    print('Initializing Model. From ' + datafile + ' load:')
    
    print(p.keys())
    
    p['hier_vals'] = p['hier_vals']/max(p['hier_vals'])    
    p['n_area'] = len(p['areas'])
      
    return p

#set the network parameters and generate the connectiivty matrix W in the linear dynamical system 
def genetate_net_connectivity(p_t,ZERO_FLN=0):
    
    p=p_t.copy()
    
   
    # disconnect all the inter-area connections 
    if ZERO_FLN:  
        p['fln_mat']=np.zeros_like(p['fln_mat'])
        print('ZERO_FLN \n')
        
#---------------------------------------------------------------------------------
# Network Parameters
#---------------------------------------------------------------------------------
    p['beta_exc'] = 0.066  # Hz/pA
    p['beta_inh'] = 0.351  # Hz/pA
    p['tau_exc'] = 20  # ms
    p['tau_inh'] = 10  # ms
    p['wEE'] = 24.4  # pA/Hz
    p['wIE'] = 12.2  # pA/Hz
    p['wEI'] = 19.7  # pA/Hz
    p['wII'] = 12.5  # pA/Hz 
    p['muEE']=33.7   # pA/Hz  
    p['muIE'] = 25.5   # pA/Hz  25.3  or smaller delta set 25.5
    p['eta'] = 0.68
    
        
    p['exc_scale'] = (1+p['eta']*p['hier_vals'])
    
    local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
    local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
    
    fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
        
   
    #---------------------------------------------------------------------------------
    # compute the connectivity matrix
    #---------------------------------------------------------------------------------
    W=np.zeros((2*p['n_area'],2*p['n_area']))       
    
    for i in range(p['n_area']):

        W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
        W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
        W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
        W[2*i+1,2*i]=local_IE[i]/p['tau_inh']
        
        W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
        W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled[i,:]/p['tau_inh']
            
    return p, W

        
#eig mode decomposition of the connectivity matrix
def eig_decomposition(p_t,W_t,EI_REARRANGE=1,CLOSE_FIG=0):
    p=p_t.copy()
    W=W_t.copy()
    
    if EI_REARRANGE==1:
        W_EI=np.zeros_like(W)
        W_EI[0:p['n_area'],0:p['n_area']]=W.copy()[0::2,0::2]
        W_EI[0:p['n_area'],p['n_area']:]=W.copy()[0::2,1::2]
        W_EI[p['n_area']:,0:p['n_area']]=W.copy()[1::2,0::2]
        W_EI[p['n_area']:,p['n_area']:]=W.copy()[1::2,1::2]
    else:
        W_EI=W
    
    #---------------------------------------------------------------------------------
    # eigenmode decomposition
    #--------------------------------------------------------------------------------- 
    eigVals, eigVecs = np.linalg.eig(W_EI)
        
    eigVecs_a=np.abs(eigVecs)
    
    tau=-1/np.real(eigVals)
    tau_s=np.zeros_like(tau)
    for i in range(len(tau)):
        tau_s[i]=format(tau[i],'.2f')
    
    ind=np.argsort(-tau_s)
    eigVecs_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))
    eigVecs_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
    tau_reorder=np.zeros(2*p['n_area'])
    
    for i in range(2*p['n_area']):
        eigVecs_a_reorder[:,i]=eigVecs_a[:,ind[i]]
        eigVecs_reorder[:,i]=eigVecs[:,ind[i]]
        tau_reorder[i]=tau_s[ind[i]]
     
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eigVecs_a_reorder),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('W matrix eigenvector visualization')
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(tau_reorder[::1])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    #fig.savefig('result/full_eigenmode.pdf')    

    #---------------------------------------------------------------------------------
    # get the slowest eigvalue
    #--------------------------------------------------------------------------------- 
    
    eigVecs_slow=eigVecs_a_reorder[:p['n_area'],:p['n_area']]
    eigVecs_slow=normalize_matrix(eigVecs_slow,column=1)
    tau_slow=tau_reorder[:p['n_area']]
        
    fig, ax = plt.subplots()
    
    f=ax.pcolormesh(eigVecs_slow,cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.15)
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    x = np.arange(len(tau_slow)) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,len(tau_slow))
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::5])
    ax.invert_xaxis()
       
    ax.set_xticklabels(tau_slow[::5])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    ax.set_title('slow eigenvector visualization')
    
    if CLOSE_FIG==1:
        plt.close('all')
        
    #---------------------------------------------------------------------------------
    # run the simulation to check the response at each area
    #---------------------------------------------------------------------------------
     
    #plt_white_noise_input(p,VISUAL_INPUT=1)

    theoretical_time_constant_input_at_one_area(p,eigVecs,eigVals,input_loc='V1')
    #theoretical_time_constant_input_at_all_areas(p,eigVecs,eigVals)  
        
    return eigVecs_a_reorder, tau_reorder

#perturbation analysis of the model
def eigen_structure_approximation(p_t):

    p=p_t.copy()
    
    _,W0=genetate_net_connectivity(p,ZERO_FLN=1)
    p,W1=genetate_net_connectivity(p,ZERO_FLN=0)
    
    theta=p['beta_exc']*p['muEE']/p['tau_exc']/(p['beta_inh']*p['muIE']/p['tau_inh'])
    print('theta=',theta)
    
    #---------------------------------------------------------------------------------
    #reshape the connectivity matrix by E and I population blocks, EE, EI, IE, II
    #---------------------------------------------------------------------------------
    W0_EI=np.zeros_like(W0)
    W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
    W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
    W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
    W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
    
    W1_EI=np.zeros_like(W1)
    W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
    W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
    W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
    W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
    
    #the variable names are consistent with symbols used in the notes
    D=W0_EI
    F=W1_EI-W0_EI
    
    D_EE=W0_EI[0:p['n_area'],0:p['n_area']]
    D_IE=W0_EI[p['n_area']:,0:p['n_area']]
    D_EI=W0_EI[0:p['n_area'],p['n_area']:]
    D_II=W0_EI[p['n_area']:,p['n_area']:]
    
    F_EE=F[0:p['n_area'],0:p['n_area']]
    F_IE=F[p['n_area']:,0:p['n_area']]
    
    cand_dei=-np.diag(D_EE)/np.diag(D_IE)*np.diag(D_II)*p['tau_exc']/p['beta_exc']
    
    fig,ax = plt.subplots()
    ax.plot(np.arange(p['n_area']),cand_dei,'-o')
    ax.set_title('for choosing wEI candidate')
    ax.set_xlabel('wEI index')
    ax.set_ylabel('wEI value that gives zero eigenvals')
    print('min wEI=',max(cand_dei))
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(D_EE,cmap='bwr')
    ax[0,0].set_title('D_EE')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(D_IE,cmap='bwr')
    ax[0,1].set_title('D_IE')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh(F_EE,cmap='bwr')
    ax[0,2].set_title('F_EE')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    f=ax[1,0].pcolormesh(D_EI,cmap='bwr')
    ax[1,0].set_title('D_EI')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(D_II,cmap='bwr')
    ax[1,1].set_title('D_II')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(F_IE,cmap='bwr')
    ax[1,2].set_title('F_IE')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    #--------------------------------------------------------------------------
    #approximations of A and B (see notes for detailed derivations)
    #--------------------------------------------------------------------------
    A=np.zeros_like(D_EE)
    A_app=np.zeros_like(A)
    B=np.zeros_like(A)
    B_app=np.zeros_like(A)
    
    for i in np.arange(p['n_area']):
        A[i,i]=0.5/D_IE[i,i]*(D_II[i,i]-D_EE[i,i]+np.sqrt((D_EE[i,i]+D_II[i,i])**2-4*(D_EE[i,i]*D_II[i,i]-D_EI[i,i]*D_IE[i,i])))
        A_app[i,i]=-D_EI[i,i]/D_II[i,i]
        B[i,i]=-D_IE[i,i]/(D_EE[i,i]+2*D_IE[i,i]*A[i,i]-D_II[i,i])
        B_app[i,i]=D_IE[i,i]/D_II[i,i]
        
    print('mean_A=',np.mean(np.diag(A)))
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(A,cmap='hot_r')
    ax[0,0].set_title('A')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(A_app,cmap='hot_r')
    ax[0,1].set_title('A_app')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh(A-A_app,cmap='hot_r')
    ax[0,2].set_title('A-A_app')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    f=ax[1,0].pcolormesh(B,cmap='hot_r')
    ax[1,0].set_title('B')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(B_app,cmap='hot_r')
    ax[1,1].set_title('B_app')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(B-B_app,cmap='hot_r')
    ax[1,2].set_title('B-B_app')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    #--------------------------------------------------------------------------
    #approximations of eigenvalues
    #--------------------------------------------------------------------------
    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.arange(p['n_area']),np.diag(D_EE+A@D_IE),'o',label='real')
    ax[0].plot(np.arange(p['n_area']),np.diag(D_EE+A_app@D_IE),'-',label='approximated')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Eigenvalue')
    ax[0].legend()
    ax[1].plot(np.arange(p['n_area']),np.diag(D_II-A@D_IE),'o',label='real')
    ax[1].plot(np.arange(p['n_area']),np.diag(D_II-A_app@D_IE),'-',label='approximated')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Eigenvalue')
    ax[1].legend()
    
    #--------------------------------------------------------------------------
    #compute P to diagnalize the local connectivity matrix without long-range connectivity
    #--------------------------------------------------------------------------
    P=np.zeros((2*p['n_area'],2*p['n_area']))
    P[0:p['n_area'],0:p['n_area']]=np.eye(p['n_area'])
    P[0:p['n_area'],p['n_area']:]=A
    P[p['n_area']:,0:p['n_area']]=B
    P[p['n_area']:,p['n_area']:]=np.eye(p['n_area'])+A@B
    P_inv=np.linalg.inv(P)
    
    fig, ax = plt.subplots(1,2)
    f=ax[0].pcolormesh(P,cmap='bwr')
    ax[0].set_title('P')
    ax[0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0],pad=0.1)
    f=ax[1].pcolormesh(P_inv,cmap='bwr')
    ax[1].set_title('P_inv')
    ax[1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1],pad=0.1)
    
    #--------------------------------------------------------------------------
    #similarity transform on the connectivity matrix using P
    #--------------------------------------------------------------------------
    Lambda=P@D@P_inv
    Sigma=P@F@P_inv
    Lambda[np.abs(Lambda)<1e-12]=0
    Sigma[np.abs(Sigma)<1e-12]=0
    
    Gamma=Lambda+Sigma
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(Lambda,cmap='bwr')
    ax[0,0].set_title('Lambda')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(Sigma,cmap='bwr')
    ax[0,1].set_title('Sigma')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh(Gamma,cmap='bwr')
    ax[0,2].set_title('Gamma')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    f=ax[1,0].pcolormesh(np.abs(Lambda),cmap='hot_r',norm=LogNorm(vmin=None,vmax=None))
    ax[1,0].set_title('|Lambda|')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(np.abs(Sigma),cmap='hot_r',norm=LogNorm(vmin=None,vmax=None))
    ax[1,1].set_title('|Sigma|')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(np.abs(Gamma),cmap='hot_r',norm=LogNorm(vmin=None,vmax=None))
    ax[1,2].set_title('|Gamma|')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    
    #--------------------------------------------------------------------------
    #extract block matrices after similarity transformation on the connectivity matrix
    #--------------------------------------------------------------------------
    Sigma_1=Sigma[0:p['n_area'],0:p['n_area']]
    Sigma_2=Sigma[0:p['n_area'],p['n_area']:]
    Sigma_3=Sigma[p['n_area']:,0:p['n_area']]
    Sigma_4=Sigma[p['n_area']:,p['n_area']:]
    Lambda_1=Lambda[0:p['n_area'],0:p['n_area']]
    Lambda_4=Lambda[p['n_area']:,p['n_area']:]
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(F_EE+A@F_IE,cmap='bwr')
    ax[0,0].set_title('F_EE+A@F_IE')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(np.eye(p['n_area'])+A@B,cmap='bwr')
    ax[0,1].set_title('I+A@B')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh((F_EE+A@F_IE)@(np.eye(p['n_area'])+A@B),cmap='bwr')
    ax[0,2].set_title('Sigma1')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    
    f=ax[1,0].pcolormesh(F_EE,cmap='bwr')
    ax[1,0].set_title('F_EE')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(A,cmap='bwr')
    ax[1,1].set_title('A')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(A@F_IE,cmap='bwr')
    ax[1,2].set_title('A@F_IE')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    #--------------------------------------------------------------------------
    #estimate the order of elements in block matrices
    #--------------------------------------------------------------------------
    Sigma_1_flat=Sigma_1.flatten()
    Sigma_1_sort=np.sort(np.abs(Sigma_1_flat))
    Sigma_2_flat=Sigma_2.flatten()
    Sigma_2_sort=np.sort(np.abs(Sigma_2_flat))
    Sigma_3_flat=Sigma_3.flatten()
    Sigma_3_sort=np.sort(np.abs(Sigma_3_flat))
    Sigma_4_flat=Sigma_4.flatten()
    Sigma_4_sort=np.sort(np.abs(Sigma_4_flat))
    
    Lambda_1_flat=Lambda_1.flatten()
    Lambda_1_sort=np.sort(np.abs(Lambda_1_flat))
    Lambda_4_flat=Lambda_4.flatten()
    Lambda_4_sort=np.sort(np.abs(Lambda_4_flat))
    
    num=np.arange(np.size(Sigma_1_sort))
    num_lbd=np.arange(p['n_area'])
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].plot(num,Sigma_1_sort,'-o')
    ax[0,0].set_title('Sigma_1_sort')
    ax[0,0].set_yscale('log')
    ax[0,0].set_ylim([0,1])
    ax[0,0].set_ylabel('element value')
    f=ax[0,1].plot(num,Sigma_2_sort,'-o')
    ax[0,1].set_title('Sigma_2_sort')
    ax[0,1].set_yscale('log')
    ax[0,1].set_ylim([0,1])
    f=ax[0,2].plot(num_lbd,Lambda_1_sort[Lambda_1_sort>0],'-o')
    ax[0,2].set_title('Lambda_1_sort')
    ax[0,2].set_yscale('log')
    ax[0,2].set_ylim([0,1])
    f=ax[1,0].plot(num,Sigma_3_sort,'-o')
    ax[1,0].set_title('Sigma_3_sort')
    ax[1,0].set_yscale('log')
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_ylabel('element value')
    f=ax[1,1].plot(num,Sigma_4_sort,'-o')
    ax[1,1].set_title('Sigma_4_sort')
    ax[1,1].set_yscale('log')
    ax[1,1].set_ylim([0,1])
    f=ax[1,2].plot(num_lbd,Lambda_4_sort[Lambda_4_sort>0],'-o')
    ax[1,2].set_title('Lambda_4_sort')
    ax[1,2].set_yscale('log')
    ax[1,2].set_ylim([1e-1,1])
    
    fig, ax = plt.subplots(1,3)
    diff_lambda1=np.zeros_like(Lambda_1)
    for i in np.arange(p['n_area']):
        for j in np.arange(p['n_area']):
           diff_lambda1[i,j]=Lambda_1[i,i]-Lambda_1[j,j]
           
    f=ax[0].plot(np.sort(diff_lambda1.flatten()),'-ro',markersize=1)
    ax[0].set_title(r'$diff\ \ \Lambda_1$')
    ax[0].set_yscale('log')
    ax[0].set_ylim([1e-5,0.1])
    ax[0].set_ylabel('element value')
    ax[0].set_xlabel('element index')
    
    f=ax[1].plot(num,Sigma_1_sort,'-go',markersize=1)
    ax[1].set_title(r'$\Sigma_1$')
    ax[1].set_yscale('log')
    ax[1].set_ylim([1e-5,0.1])
    ax[1].set_xlabel('element index')
    
    f=ax[2].plot(num,Sigma_2_sort,'-bo',markersize=1)
    ax[2].set_title(r'$\Sigma_2$')
    ax[2].set_yscale('log')
    ax[2].set_ylim([1e-5,0.1])
    ax[2].set_xlabel('element index')
      
    fig, ax = plt.subplots()
    ax.plot(np.sort(np.abs(np.real(diff_lambda1.flatten()))),'o',markersize=3,label=r'$\Lambda_1$')
    ax.plot(num,Sigma_1_sort,'o',markersize=3,label=r'$\Sigma_1$')
    ax.plot(num,Sigma_2_sort,'o',markersize=3,label=r'$\Sigma_2$')
    ax.set_yscale('log')
    ax.set_ylim([0,1])
    ax.set_ylabel('values')
    ax.set_xlabel('index')
    ax.legend()

    
    
    #--------------------------------------------------------------------------
    #compare the gap of eigenvalues and the off-diagnal elements
    #--------------------------------------------------------------------------
    temp_lbd=np.diag(Lambda_1)
    dif_lbd=np.zeros(p['n_area']**2)
    count=0
    for i in np.arange(p['n_area']):
        for j in np.arange(i+1,p['n_area']):
            dif_lbd[count]=np.abs(temp_lbd[i]-temp_lbd[j])
            count=count+1
            
    fig, ax = plt.subplots()
    #f=ax.hist(np.diff(np.diag(Lambda_1)),200,facecolor='r')
    f=ax.hist(dif_lbd[dif_lbd>0],100,facecolor='r',alpha=0.5,label='eigval diff')
    Sigma_nonzero=Sigma_1.flatten()
    Sigma_nonzero=Sigma_nonzero[Sigma_nonzero>0]
    f=ax.hist(Sigma_nonzero,200,facecolor='b',alpha=0.5,label='sigma_1')
    ax.set_yscale('log')
    ax.set_xlabel('value')
    ax.set_ylabel('counts')
    ax.legend()
    print('mean_sigma1=',np.mean(Sigma_nonzero))
#------------------------------------------------------------------------------
#approximation of the eigenvectors with components u and v
#------------------------------------------------------------------------------
    
#------------------------------first u case------------------------------------
    #eigvals_norder, eigvecs_norder = np.linalg.eig(Lambda_1+Sigma_1)    #TEST TEST TEST np.linalg.eig(Lambda_1+Sigma_1)
    
    # ind=np.argsort(-np.real(eigvals_norder))
    # id_mat=np.eye(p['n_area'])
    # per_mat=id_mat.copy()
    # for i in np.arange(p['n_area']):
    #     per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    # eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    # eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    # eigvals=np.diag(eigmat) 
    
    # u_1=eigvecs.copy()
    # v_1=-np.linalg.inv(Lambda_4+Sigma_4-np.diag(eigvals))@Sigma_3@eigvecs
    # lbd_1=eigvals.copy()
    
    # r_E_1=(np.eye(p['n_area'])+A@B)@u_1-A@v_1
    # r_I_1=-B@u_1+v_1

    #2021-3-20 compute the perturbated eigenvalues and eigenvectors of Lambda_1 + Sigma_1 TEST TEST TEST 2021-3-20
    eigvals_norder, sec_order_eigvals,eigvecs_norder,sec_order_eigvecs=eig_approx(Lambda_1,Sigma_1)
    ind=np.argsort(-np.real(eigvals_norder))
    id_mat=np.eye(p['n_area'])
    per_mat=id_mat.copy()
    for i in np.arange(p['n_area']):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    eigvals=np.diag(eigmat) 
    
    u_1=eigvecs.copy()
    v_1=-np.linalg.inv(Lambda_4+Sigma_4-np.diag(eigvals))@Sigma_3@eigvecs
    lbd_1=eigvals.copy()
    
    r_E_1=(np.eye(p['n_area'])+A@B)@u_1-A@v_1
    r_I_1=-B@u_1+v_1
    

#------------------------------second u case-----------------------------------    
    eigvals_norder, eigvecs_norder = np.linalg.eig(Lambda_4+Sigma_4)
    
    ind=np.argsort(-np.real(eigvals_norder))
    id_mat=np.eye(p['n_area'])
    per_mat=id_mat.copy()
    for i in np.arange(p['n_area']):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    eigvals=np.diag(eigmat) 

    u_2=np.zeros((p['n_area'],p['n_area']))
    v_2=eigvecs.copy()
    lbd_2=eigvals.copy()
    
    r_E_2=(np.eye(p['n_area'])+A@B)@u_2-A@v_2
    r_I_2=-B@u_2+v_2
    
#-----------------------------plot u and v-------------------------------------  
    eig_val=(0+0j)*np.zeros(2*p['n_area'])
    eig_val[0:p['n_area']]=lbd_1
    eig_val[p['n_area']:]=lbd_2
     
    eig_vec=(0+0j)*np.zeros((2*p['n_area'],2*p['n_area']))
    eig_vec[0:p['n_area'],0:p['n_area']]=u_1
    eig_vec[p['n_area']:,0:p['n_area']]=v_1
    eig_vec[0:p['n_area'],p['n_area']:]=u_2
    eig_vec[p['n_area']:,p['n_area']:]=v_2
    
    for i in np.arange(2*p['n_area']):
        eig_vec[:,i]=eig_vec[:,i]/np.linalg.norm(eig_vec[:,i])
        
    temp_eig=np.real(eig_val)
    eigvals_s=np.zeros_like(temp_eig)
    for i in range(len(temp_eig)):
        eigvals_s[i]=format(temp_eig[i],'.5f')
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eig_vec),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('eigenvector visualization-u-v')

    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
      
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_xticks(x[::1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    ax.set_xticklabels(eigvals_s[::1])
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    
#--------------------------------plot R_E R_I----------------------------------   
    eig_val=(0+0j)*np.zeros(2*p['n_area'])
    eig_val[0:p['n_area']]=lbd_1
    eig_val[p['n_area']:]=lbd_2
     
    eig_vec=(0+0j)*np.zeros((2*p['n_area'],2*p['n_area']))
    eig_vec[0:p['n_area'],0:p['n_area']]=r_E_1
    eig_vec[p['n_area']:,0:p['n_area']]=r_I_1
    eig_vec[0:p['n_area'],p['n_area']:]=r_E_2
    eig_vec[p['n_area']:,p['n_area']:]=r_I_2
    
    for i in np.arange(2*p['n_area']):
        eig_vec[:,i]=eig_vec[:,i]/np.linalg.norm(eig_vec[:,i])
        
    temp_eig=np.real(eig_val)
    eigvals_s=np.zeros_like(temp_eig)
    for i in range(len(temp_eig)):
        eigvals_s[i]=format(temp_eig[i],'.5f')
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eig_vec),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('eigenvector visualization_rE_rI')

    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
      
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_xticks(x[::1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    ax.set_xticklabels(eigvals_s[::1])
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    
    #--------------------------------------------------------------------------
    #eigenmode decomposition
    #--------------------------------------------------------------------------
    eigvals_norder, eigvecs_norder = np.linalg.eig(W1_EI)
    ind=np.argsort(-np.real(eigvals_norder))
    id_mat=np.eye(2*p['n_area'])
    per_mat=id_mat.copy()
    for i in np.arange(2*p['n_area']):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    eigvals=np.diag(eigmat) 
        
    temp_eig=np.real(eigvals)
    eigvals_s=np.zeros_like(temp_eig)
    for i in range(len(temp_eig)):
        eigvals_s[i]=format(temp_eig[i],'.5f')
        
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eigvecs),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('W matrix eigenvector visualization')
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(eigvals_s[::1])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   


    #--------------------------------------------------------------------------
    #egienmatrix approximation resort 2021-3-20
    #--------------------------------------------------------------------------
    eigvecs_ori=np.abs(eigvecs)  #from original
    eigvecs_app=np.abs(eig_vec)    #from approximation
    index_list=[]
    max_inprod_list=[]
    
    length=len(temp_eig)
    for i in range(length):
        max_inprod=-1
        for j in range(length):
            # print('j=',j)
            # print(index_list)
            # j not in index_list
            if j not in index_list: 
               temp_max_inprod = np.dot(eigvecs_ori[:,length-i-1], eigvecs_app[:,length-j-1])
               if temp_max_inprod >max_inprod:
                   max_inprod=temp_max_inprod
                   max_index=j
        max_inprod_list.append(max_inprod)
        index_list.append(max_index)
    print(index_list)    
    print(max_inprod_list)
    
    plt.figure(figsize=(20,5))        
    ax = plt.axes()
    plt.bar(np.arange(length),max_inprod_list,color='k',alpha=0.5)
    # plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('similarity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    eigvecs_reorder=np.zeros_like(eigvecs_app)
    for i in np.arange(2*p['n_area']):        
        eigvecs_reorder[:,length-i-1]=eigvecs_app[:,length-1-index_list[i]]
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eigvecs_reorder),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('W approximation eigenvector visualization reorder')
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(eigvals_s[::1])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
      
        
#run simulation of netowrk responses    
def run_stimulus(p_t,VISUAL_INPUT=1,PULSE_INPUT=1,MACAQUE_CASE=1,GATING_PATHWAY=0,CONSENSUS_CASE=0):
        
        if VISUAL_INPUT:
            area_act = 'V1'   #V1
        else:
            if MACAQUE_CASE:
                area_act='2'
            else:
                area_act = 'AuA1'
        print('Running network with stimulation to ' + area_act + '   PULSE_INPUT=' + str(PULSE_INPUT) + '   MACAQUE_CASE=' + str(MACAQUE_CASE))

        #---------------------------------------------------------------------------------
        # Redefine Parameters
        #---------------------------------------------------------------------------------

        p=p_t.copy()

        # Definition of combined parameters

        local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] * np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
        local_II = -p['beta_inh'] * p['wII'] * np.ones_like(local_EE)
        
    
        fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
            
        #---------------------------------------------------------------------------------
        # Simulation Parameters
        #---------------------------------------------------------------------------------

        dt = 0.05   # ms
        if PULSE_INPUT:
            T = 1600
        else:                
            T = 2e5
        
        t_plot = np.linspace(0, T, int(T/dt)+1)
        n_t = len(t_plot)  

        # From target background firing inverts background inputs
        r_exc_tgt = 10 * np.ones(p['n_area'])    
        r_inh_tgt = 35 * np.ones(p['n_area'])

        longrange_E = np.dot(fln_scaled,r_exc_tgt)
        I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                                 + p['beta_exc']*p['muEE']*longrange_E)
        I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                                 + p['beta_inh']*p['muIE']*longrange_E)

        # Set stimulus input
        I_stim_exc = np.zeros((n_t,p['n_area']))

        area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
        
        # #==============TEST TEST TEST XXXXX DEBUG DEBUG=========================
        # area_stim_idx2 = p['areas'].index('9/46d') # Index of stimulated area
        
        if PULSE_INPUT:
            time_idx = (t_plot>100) & (t_plot<=350)
            I_stim_exc[time_idx, area_stim_idx] = 41.187
            # I_stim_exc[time_idx, area_stim_idx2] = 41.187
        else:
            for i in range(p['n_area']):
                I_stim_exc[:,i]=gaussian_noise(0,1e-5,n_t)        
            I_stim_exc[:, area_stim_idx] = gaussian_noise(2,0.5,n_t) #2,0.5
        
        #---------------------------------------------------------------------------------
        # Storage
        #---------------------------------------------------------------------------------

        r_exc = np.zeros((n_t,p['n_area']))
        r_inh = np.zeros((n_t,p['n_area']))

        #---------------------------------------------------------------------------------
        # Initialization
        #---------------------------------------------------------------------------------
        fI = lambda x : x*(x>0)
        #fI = lambda x : x*(x>0)*(x<300)+300*(x>300)  #for GATING_PATHWAY==1 only
        
        # Set activity to background firing
        r_exc[0] = r_exc_tgt
        r_inh[0] = r_inh_tgt
        
        #---------------------------------------------------------------------------------
        # Running the network
        #---------------------------------------------------------------------------------

        for i_t in range(1, n_t):
            longrange_E = np.dot(fln_scaled,r_exc[i_t-1])
            I_exc = (local_EE*r_exc[i_t-1] + local_EI*r_inh[i_t-1] +
                     p['beta_exc'] * p['muEE'] * longrange_E +
                     I_bkg_exc + I_stim_exc[i_t])

            I_inh = (local_IE*r_exc[i_t-1] + local_II*r_inh[i_t-1] +
                     p['beta_inh'] * p['muIE'] * longrange_E + I_bkg_inh)
            
            if GATING_PATHWAY:
                d_local_EI=np.zeros_like(local_EI)
                if MACAQUE_CASE:
                    if VISUAL_INPUT:
                        area_name_list = ['V4','8m']
                 
                for name in area_name_list:
                    area_idx=p['areas'].index(name)
                    d_local_EI[area_idx]=-local_EI[area_idx]*0.07  #0.1
                
                if I_stim_exc[i_t,area_stim_idx]>10:
                    I_exc=I_exc+d_local_EI*r_inh[i_t-1]
                
            d_r_exc = -r_exc[i_t-1] + fI(I_exc)
            d_r_inh = -r_inh[i_t-1] + fI(I_inh)

            r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt/p['tau_exc']
            r_inh[i_t] = r_inh[i_t-1] + d_r_inh * dt/p['tau_inh']
        
        #---------------------------------------------------------------------------------
        # Plotting step input results
        #---------------------------------------------------------------------------------
        if CONSENSUS_CASE==0:
            if MACAQUE_CASE:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V4','8m','8l','TEO','7A','9/46d','TEpd','24c']
                else:
                    area_name_list = ['5','2','F1','10','9/46v','9/46d','F5','7B','F2','ProM','F7','8B','24c']
            else:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
                else:
                    area_name_list = ['AuA1','A1-2','V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
        else:
            if MACAQUE_CASE:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V4','8m 8l 8r','5','TEO TEOm','F4','9/46d 46d','TEpd TEa/ma TEa/mp','F7']
                else:
                    raise SystemExit('Must give Visual input to networks under consensus map!')
                    
            else:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V2','V4','PEC PE','LIP','PGM','A32 A32V','A6DR','A6Va A6Vb']
                else:
                    raise SystemExit('Must give Visual input to networks under consensus map!')
                    
                    
                    
        area_idx_list=[-1]
        for name in area_name_list:
            area_idx_list=area_idx_list+[p['areas'].index(name)]
        #area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
        
        f, ax_list = plt.subplots(len(area_idx_list), sharex=True)
        
        clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
        c_color=0
        for ax, area_idx in zip(ax_list, area_idx_list):
            if area_idx < 0:
                y_plot = I_stim_exc[:, area_stim_idx].copy()
                z_plot = np.zeros_like(y_plot)
                txt = 'Input'

            else:
                y_plot = r_exc[:,area_idx].copy()
                z_plot = r_inh[:,area_idx].copy()
                txt = p['areas'][area_idx]

            if PULSE_INPUT:
                y_plot = y_plot - y_plot.min()
                z_plot = z_plot - z_plot.min()
                ax.plot(t_plot, y_plot,color='k')
                #ax.plot(t_plot, z_plot,'--',color='b')
            else:
                #ax.plot(t_plot, y_plot,color='r')
                ax.plot(t_plot[0:10000], y_plot[-1-10000:-1],color='r')
                # ax.plot(t_plot[0:10000], z_plot[-1-10000:-1],'--',color='b')
                
            # ax.plot(t_plot, y_plot,color=clist[0][c_color])
            # ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
            c_color=c_color+1
            ax.text(0.9, 0.6, txt, transform=ax.transAxes)

            if PULSE_INPUT:
                ax.set_yticks([0,y_plot.max()])
                ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            #ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
        ax.set_xlabel('Time (ms)')    
               
        return I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot
 

#simulate the network by giving white noise input and estimate the time constant by fitting the autocorrelation function
def plt_white_noise_input(p_t,VISUAL_INPUT=1):
    
    print('In function: plt_white_noise_input, parameters are VISUAL_INPUT='+str(VISUAL_INPUT))
    
    NORMAL_case=1
            
    p = p_t.copy()
    I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot=run_stimulus(p,VISUAL_INPUT=VISUAL_INPUT,PULSE_INPUT=0)
    
    #area_name_list = ['V1','V4','8m','8l','TEO','2','7A','10','9/46v','9/46d','TEpd','7m','7B','24c']
    area_name_list=p['areas']
    area_idx_list=[-1]
    
    T_lag=int(3e3)
    acf_data=np.zeros((len(area_name_list),T_lag+1))   
    
    for name in area_name_list:
            area_idx_list=area_idx_list+[p['areas'].index(name)]
            
    f, ax_list = plt.subplots(len(area_idx_list), sharex=True,figsize=(10,15))
    j=0
    for ax, area_idx in zip(ax_list, area_idx_list):
        if area_idx < 0:
            y_plot = I_stim_exc[::int(1/dt), area_stim_idx].copy()
            txt = 'Input'
            
        else:
            y_plot = r_exc[::int(1/dt),area_idx].copy()
            txt = p['areas'][area_idx]
            acf_data[j,:] = smt.stattools.acf(y_plot,nlags=T_lag, fft=True)
            j=j+1
            
        y_plot = y_plot - y_plot.min()
        ax.plot(t_plot[::int(1/dt)], y_plot)
        ax.text(0.9, 0.6, txt, transform=ax.transAxes)

        ax.set_yticks([y_plot.max()])
        ax.set_yticklabels(['{:0.4f}'.format(y_plot.max())])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')

    f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
    ax.set_xlabel('Time (ms)')
                
    clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_name_list)))[np.newaxis, :, :3]

    for name in area_name_list:
        area_idx_list=area_idx_list+[p['areas'].index(name)]
    
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,20))
    for i in np.arange(len(area_name_list)):
        ax1.plot(np.arange(T_lag+1),acf_data[i,:],color=clist[0][i])
        
    ax1.set_xlabel('Time difference (ms)')
    ax1.set_ylabel('Correlation')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #plt.savefig('result/correlation_stim_V1.pdf')
    
     
    #---------------------------------------------------------------------------------
    # parameter fit
    #---------------------------------------------------------------------------------    
    t_plot=t_plot[::int(1/dt)].copy()
     
    delay_time=np.zeros(len(area_name_list))
    f, ax_list = plt.subplots(len(area_name_list), sharex=True, figsize=(15,15))
    for ax, i in zip(ax_list, np.arange(len(area_name_list))):
        p_end=np.where(acf_data[i,:]>0.05)[0][-1]
        
        r_single, _ =optimize.curve_fit(single_exp,t_plot[0:p_end],acf_data[i,0:p_end])
        r_double, _ =optimize.curve_fit(double_exp,t_plot[0:p_end],acf_data[i,0:p_end],p0=[r_single[0],0.1,r_single[1],0],bounds=(0,np.inf),maxfev=5000)
        
        e_single=sum((acf_data[i,0:p_end]-r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]))**2)
        e_double=sum((acf_data[i,0:p_end]-(r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1])))**2)
        
        e_ratio=e_single/e_double
        
        if e_ratio>8:
            delay_time[i]=r_double[0]
        else:
            delay_time[i]=r_single[0]
                
        #print('error ratio of',area_name_list[i],"=",str(e_ratio))
        
        ax.plot(t_plot[0:p_end],acf_data[i,0:p_end])
        ax.plot(t_plot[0:p_end],r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]),'r--')
        ax.plot(t_plot[0:p_end],r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1]),'g--')
        ax.set_ylim(0,1)
        txt = area_name_list[i]
        ax.text(0.9, 0.6, txt, transform=ax.transAxes)
        
    f.text(0.01, 0.5, 'Simulated correlation', va='center', rotation='vertical')
    ax.set_xlabel('Time difference (ms)')
  
    ax2.bar(np.arange(len(area_name_list)),delay_time,width = 1,color=clist[0])
    ax2.set_xticks(np.arange(len(area_name_list)))
    ax2.set_xticklabels(area_name_list,rotation=90)
    ax2.set_yticks([10,100,1000])
    ax2.set_yticklabels(['10 ms','100 ms','1 s'],rotation=0)
    ax2.set_ylabel('$Simulated T_{delay}$ (ms)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ind=np.argsort(delay_time)
    delay_time_sort=np.zeros_like(delay_time)
    area_name_list_sort=[]
    for i in np.arange(len(ind)):
        delay_time_sort[i]=delay_time[ind[i]]
        area_name_list_sort=area_name_list_sort+[area_name_list[ind[i]]]
        
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time_sort,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list_sort,rotation=90)
    plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('$Simulated T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if NORMAL_case:
        plt.savefig('result/normal.pdf')
        
    
#fitting the time constants of each area from its theoretical autocorrelation function when giving an input at a specific input location     
def theoretical_time_constant_input_at_one_area(p_t,eigVecs,eigVals,input_loc='V1'):
    p=p_t.copy()
    area_name_list=p['areas']
    
    inv_eigVecs=np.linalg.inv(eigVecs)
    
    n=len(area_name_list)
    T_lag=int(3e3)
    
    m=p['areas'].index(input_loc)         
    acf_data=np.zeros((n,T_lag+1))+0j 
    coef=np.zeros(2*n)+0j

    for i in np.arange(n):
        for s in np.arange(T_lag+1):
            for j in np.arange(2*n):
                coef[j]=0
                for k in np.arange(2*n):
                    coef[j]=coef[j]+eigVecs[i,j]*eigVecs[i,k]*inv_eigVecs[j,m]*inv_eigVecs[k,m]/(-eigVals[j]-eigVals[k])
                acf_data[i,s] = acf_data[i,s]+coef[j]*np.exp(eigVals[j]*s)
        acf_data[i,:]=acf_data[i,:]/acf_data[i,0]
            
    clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_name_list)))[np.newaxis, :, :3]
    
    plt.figure(figsize=(10,5))
    ax = plt.axes()
    for i in np.arange(len(area_name_list)):
        plt.plot(np.arange(T_lag+1),acf_data[i,:],color=clist[0][i])
        
    plt.legend(area_name_list)
    plt.xlabel('Time difference (ms)')
    plt.ylabel('Theoretical correlation')
    plt.title('input_loc='+input_loc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.savefig('result/correlation_stim_V1.pdf')
    
    t_plot=np.arange(T_lag)
     
    delay_time=np.zeros(len(area_name_list))
    f, ax_list = plt.subplots(len(area_name_list), sharex=True, figsize=(15,15))
    for ax, i in zip(ax_list, np.arange(len(area_name_list))):
        p_end=np.where(acf_data[i,:]>0.05)[0][-1]
        
        r_single, _ =optimize.curve_fit(single_exp,t_plot[0:p_end],acf_data[i,0:p_end])
        r_double, _ =optimize.curve_fit(double_exp,t_plot[0:p_end],acf_data[i,0:p_end],p0=[r_single[0],0.1,r_single[1],0],bounds=(0,np.inf),maxfev=5000)
        
        e_single=sum((acf_data[i,0:p_end]-r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]))**2)
        e_double=sum((acf_data[i,0:p_end]-(r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1])))**2)
        
        e_ratio=e_single/e_double
        
        if e_ratio>8:
            delay_time[i]=r_double[0]
        else:
            delay_time[i]=r_single[0]
                
        #print('error ratio of',area_name_list[i],"=",str(e_ratio))
        
        ax.plot(t_plot[0:p_end],acf_data[i,0:p_end])
        ax.plot(t_plot[0:p_end],r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]),'r--')
        ax.plot(t_plot[0:p_end],r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1]),'g--')
        ax.set_ylim(0,1)
        txt = area_name_list[i]
        ax.text(0.9, 0.6, txt, transform=ax.transAxes)
        
    f.text(0.01, 0.5, 'Theoretical Correlation', va='center', rotation='vertical')
    ax.set_xlabel('Time difference (ms)')
    ax.set_title('input_loc='+input_loc)
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list,rotation=90)
    #plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('Theoretical $T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('input_loc='+input_loc)
    
    ind=np.argsort(delay_time)
    delay_time_sort=np.zeros_like(delay_time)
    area_name_list_sort=[]
    for i in np.arange(len(ind)):
        delay_time_sort[i]=delay_time[ind[i]]
        area_name_list_sort=area_name_list_sort+[area_name_list[ind[i]]]
        
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time_sort,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list_sort,rotation=90)
    #plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('Theoretical $T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('input_loc='+input_loc)


#------------------------------------------------------------------------------
#auxiliary functions used in other functions
#------------------------------------------------------------------------------
    
# plot the relation between the E and I parts of the eigenvectors
def plot_re_ri_eigvecs(p,eigVecs_t):
    eigVecs=eigVecs_t.copy()
    lens=p['n_area']
    fig,ax=plt.subplots()
    for i in np.arange(lens):
            ax.scatter(eigVecs[2*i,lens+1:-1],eigVecs[2*i+1,lens+1:-1],label=p['areas'][i])
    ax.legend()
    ax.set_xlabel('E population rate')
    ax.set_ylabel('I population rate')
    ax.legend(loc='best',ncol=2)


def local_index(x_ori,y_ori,x_new,y_new):
    ind=np.mean(x_new)/np.mean(x_ori)*np.mean(y_new)/np.mean(y_ori)
    return ind

#generate gaussian noise    
def gaussian_noise(mu,sigma,n_t):
    input_sig=np.zeros(n_t)
    for i in range(0,n_t):
        input_sig[i]=random.gauss(mu,sigma)
    return input_sig

def single_exp(x,a,b):
    return b*np.exp(-x/a)

def double_exp(x,a,b,c,d):
    return c*np.exp(-x/a)-d*np.exp(-x/b)

#compute eigen vector and value for desired direction.
def pick_eigen_direction(egval, egvec, direction_ref, mode, ord=0):
    # Return eigen vector and value for desired direction.
    # @param egval: eigen values list.
    # @param egvec: each column is an eigen vector, normalized to 2-norm == 1.
    # @param direction_ref: reference direction, can be a vector or column vectors.
    # @param mode:
    # @param ord: to select the `ord`-th close direction. For switching direction.
    if mode == 'smallest':
        id_egmin = np.argmin(np.abs(egval))
        zero_g_direction = egvec[:, id_egmin]
        if np.dot(zero_g_direction, direction_ref) < 0:
            zero_g_direction = -zero_g_direction
        return egval[id_egmin], zero_g_direction
    elif mode == 'continue':
        # pick the direction that matches the previous one
        if direction_ref.ndim==1:
            direction_ref = direction_ref[:, np.newaxis]
        n_pick = direction_ref.shape[1]    # number of track
        vec_pick = np.zeros_like(direction_ref)
        val_pick = np.zeros_like(egval)
        similarity = np.dot(egvec.conj().T, direction_ref)
        for id_v in range(n_pick):
            # id_pick = np.argmin(np.abs(np.abs(similarity[:, id_v])-1))
            id_pick = np.argsort(np.abs(np.abs(similarity[:, id_v])-1))[ord]
            if similarity[id_pick, id_v] > 0:
                vec_pick[:, id_v] = egvec[:, id_pick]
            else:
                vec_pick[:, id_v] = -egvec[:, id_pick]
            val_pick[id_v] = egval[id_pick]
        return np.squeeze(val_pick), np.squeeze(vec_pick)
    elif mode == 'close-egval':
        # direction_ref should be pair (egval, egvec)
        old_egval = direction_ref[0]
        old_egvec = direction_ref[1]
        if len(old_egvec.shape) == 1:
            old_egvec = old_egvec[:, np.newaxis]
        egdist = np.abs(old_egval[:,np.newaxis] - egval[np.newaxis, :])
        # Greedy pick algo 1
        # 1. loop for columns of distance matrix
        # 2.    pick most close pair of eigenvalue and old-eigenvalue.
        # 3.    remove corresponding row in the distance matrix.
        # 4. go to 1.
        # Greedy pick algo 2
        # 1. pick most close pair of eigenvalue and old-eigenvalue.
        # 2. remove corresponding column and row in the distance matrix.
        # 3. go to 1.
        # Use algo 1.
        #plt.matshow(egdist)
        n_pick = old_egvec.shape[1]
        mask = np.arange(n_pick)
        vec_pick = np.zeros_like(old_egvec)
        val_pick = np.zeros_like(egval)
        #print('old_eigval=\n', old_egval[:,np.newaxis])
        #print('new eigval=\n', egval[:,np.newaxis])
        for id_v in range(n_pick):
            id_pick_masked = np.argmin(egdist[id_v, mask])
            #print('mask=',mask)
            id_pick = mask[id_pick_masked]
            #print('id_pick=',id_pick, '  eigval=', egval[id_pick])
            val_pick[id_v] = egval[id_pick]
            # might try: sign = np.exp(np.angle(...)*1j)
            if np.angle(np.vdot(egvec[:,id_pick], old_egvec[:, id_v])) > 0:
                vec_pick[:, id_v] = egvec[:, id_pick]
            else:
                vec_pick[:, id_v] = -egvec[:, id_pick]
            mask = np.delete(mask, id_pick_masked)
        return np.squeeze(val_pick), np.squeeze(vec_pick)
    else:
        raise ValueError()    
        

#compute the eigenvalues and eigenvectors of A+dA by perturbation analysis
def eig_approx(A,dA):
    
    eigval, eigvec = np.linalg.eig(A)
    id_sort = np.argsort(abs(eigval.real))
    eigval_ori=eigval[id_sort]
    eigvec_ori=eigvec[:, id_sort]
    
    eigval_l, eigvec_l = np.linalg.eig(A.T)
    id_sort_l = np.argsort(abs(eigval_l.real))
    eigvec_ori_l=eigvec_l[:, id_sort_l]
    
    eigvec_ori_left=eigvec_ori_l.conj()
    
    # print('eigval=',eigval_ori)
    # print('eigval_l=',eigval_ori_l)
    
    A_size=A.shape[0]
    d_eigval_1=np.zeros(A_size, dtype='complex128')
    d_eigval_2=np.zeros(A_size, dtype='complex128')
    d_eigvec_1=np.zeros((A_size,A_size), dtype='complex128')
    d_eigvec_2=np.zeros((A_size,A_size), dtype='complex128')

    for i in range(A_size):
        d_eigval_1[i]=eigvec_ori_left[:,i].conj().T@dA@eigvec_ori[:,i]/(eigvec_ori_left[:,i].conj().T@eigvec_ori[:,i])
        
        for k in range(A_size):
            if k!=i:
                d_eigvec_1[:,i]=d_eigvec_1[:,i]+eigvec_ori_left[:,k].conj().T@dA@eigvec_ori[:,i]*eigvec_ori[:,k]/(eigval_ori[i]-eigval_ori[k])/(eigvec_ori_left[:,k].conj().T@eigvec_ori[:,k])
                d_eigval_2[i]=d_eigval_2[i]+(eigvec_ori_left[:,k].conj().T@dA@eigvec_ori[:,i])*(eigvec_ori_left[:,i].conj().T@dA@eigvec_ori[:,k])/(eigval_ori[i]-eigval_ori[k])/(eigvec_ori_left[:,k].conj().T@eigvec_ori[:,k])/(eigvec_ori_left[:,i].conj().T@eigvec_ori[:,i])
    
        # for m in range(A_size):
        #     if m!=i:
        #         d_eigvec_2[:,i]=d_eigvec_2[:,i]+(eigvec_ori_left[:,i].T@dA@eigvec_ori[:,i])*(eigvec_ori_left[:,m].T@dA@eigvec_ori[:,i])/(eigval_ori[i]-eigval_ori[m])*eigvec_ori[:,m]/(eigval_ori[m]-eigval_ori[i])/(eigvec_ori_left[:,i].T@eigvec_ori[:,i])/(eigvec_ori_left[:,m].T@eigvec_ori[:,m])
        #         for k in range(A_size):
        #             if k!=i:
        #                 d_eigvec_2[:,i]=d_eigvec_2[:,i]-(eigvec_ori_left[:,k].T@dA@eigvec_ori[:,i])*(eigvec_ori_left[:,m].T@dA@eigvec_ori[:,k])/(eigval_ori[i]-eigval_ori[k])*eigvec_ori[:,m]/(eigval_ori[m]-eigval_ori[i])/(eigvec_ori_left[:,m].T@eigvec_ori[:,m])/(eigvec_ori_left[:,k].T@eigvec_ori[:,k])
  
    return eigval_ori+d_eigval_1, eigval_ori+d_eigval_1+d_eigval_2,eigvec_ori+d_eigvec_1,eigvec_ori+d_eigvec_1+d_eigvec_2

#compute the angle between two eigenvectors    
def normality_meaure(W_t):
    W=W_t.copy()
    n=np.shape(W)[1]
    
    for i in np.arange(n):
        W[:,i]=W[:,i]/np.linalg.norm(W[:,i])
        
    theta_mat=W.T@W
    theta_ang=np.zeros_like(theta_mat)
    for i in np.arange(n):
        for j in np.arange(n):
            if theta_mat[i,j]>1:
                theta_mat[i,j]=1
            theta_ang[i,j]=np.arccos(theta_mat[i,j])*180/np.pi
            
    fig, ax = plt.subplots(figsize=(13,10))
    f=ax.pcolormesh(theta_ang,cmap='hot_r')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('theta angle')
    ax.invert_yaxis()
    
    mean_ang=(np.sum(theta_ang)-n*0)/(n**2-n)
    print('mean_ang=',mean_ang)

    
def unstability_detection(p_t,fln_t):
    
    p=p_t.copy()
    fln=fln_t.copy()
    
    local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
    local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
        
    fln_scaled = (p['exc_scale'] * fln.T).T
    
    #---------------------------------------------------------------------------------
    # the first way to compute the connectivity matrix
    #---------------------------------------------------------------------------------
    W=np.zeros((2*p['n_area'],2*p['n_area']))       
    
    for i in range(p['n_area']):
        W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
        W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
        W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
        W[2*i+1,2*i]=local_IE[i]/p['tau_inh']

        W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
        W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled[i,:]/p['tau_inh']
        
    eigvals, eigvecs = np.linalg.eig(W)   
    
    max_eigval=np.max(eigvals)
    
    return max_eigval, W



def normalize_matrix(M_t,column=1):
    M=M_t.copy()
    r,c=M.shape
    M_norm=np.zeros_like(M)

    if column==1:
        for i in np.arange(c):
            M_norm[:,i]=M[:,i]/np.linalg.norm(M[:,i])
    else:
        for i in np.arange(r):
            M_norm[i,:]=M[i,:]/np.linalg.norm(M[i,:])
   
    return M_norm

def get_Sigma_Lambda_matrix(p_t,W_t,MACAQUE_CASE,CONSENSUS_CASE=0):
    
    p=p_t.copy()
    _,W0=genetate_net_connectivity(p,LOCAL_IDENTICAL_HIERARCHY=0,ZERO_FLN=1,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
    #---------------------------------------------------------------------------------
    #reshape the connectivity matrix by E and I population blocks, EE, EI, IE, II
    #---------------------------------------------------------------------------------
    W0_EI=np.zeros_like(W0)
    W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
    W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
    W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
    W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
    
    W1=W_t.copy()
    W1_EI=np.zeros_like(W1)
    W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
    W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
    W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
    W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
    
    #the variable names are consistent with symbols used in the notes
    D=W0_EI
    F=W1_EI-W0_EI
    
    D_EE=W0_EI[0:p['n_area'],0:p['n_area']]
    D_IE=W0_EI[p['n_area']:,0:p['n_area']]
    D_EI=W0_EI[0:p['n_area'],p['n_area']:]
    D_II=W0_EI[p['n_area']:,p['n_area']:]
    
    # F_EE=F[0:p['n_area'],0:p['n_area']]
    # F_IE=F[p['n_area']:,0:p['n_area']]
        
    #--------------------------------------------------------------------------
    #approximations of A and B (see notes for detailed derivations)
    #--------------------------------------------------------------------------
    A=np.zeros_like(D_EE)
    A_app=np.zeros_like(A)
    B=np.zeros_like(A)
    B_app=np.zeros_like(A)
    
    for i in np.arange(p['n_area']):
        A[i,i]=0.5/D_IE[i,i]*(D_II[i,i]-D_EE[i,i]+np.sqrt((D_EE[i,i]+D_II[i,i])**2-4*(D_EE[i,i]*D_II[i,i]-D_EI[i,i]*D_IE[i,i])))
        A_app[i,i]=-D_EI[i,i]/D_II[i,i]
        B[i,i]=-D_IE[i,i]/(D_EE[i,i]+2*D_IE[i,i]*A[i,i]-D_II[i,i])
        B_app[i,i]=D_IE[i,i]/D_II[i,i]
         
  
    #--------------------------------------------------------------------------
    #compute P to diagnalize the local connectivity matrix without long-range connectivity
    #--------------------------------------------------------------------------
    P=np.zeros((2*p['n_area'],2*p['n_area']))
    P[0:p['n_area'],0:p['n_area']]=np.eye(p['n_area'])
    P[0:p['n_area'],p['n_area']:]=A
    P[p['n_area']:,0:p['n_area']]=B
    P[p['n_area']:,p['n_area']:]=np.eye(p['n_area'])+A@B
    P_inv=np.linalg.inv(P)
       
    #--------------------------------------------------------------------------
    #similarity transform on the connectivity matrix using P
    #--------------------------------------------------------------------------
    Lambda=P@D@P_inv
    Sigma=P@F@P_inv
    Lambda[np.abs(Lambda)<1e-12]=0
    Sigma[np.abs(Sigma)<1e-12]=0
    
    #--------------------------------------------------------------------------
    #extract block matrices after similarity transformation on the connectivity matrix
    #--------------------------------------------------------------------------
    Sigma_1=Sigma[0:p['n_area'],0:p['n_area']]
    # Sigma_2=Sigma[0:p['n_area'],p['n_area']:]
    # Sigma_3=Sigma[p['n_area']:,0:p['n_area']]
    # Sigma_4=Sigma[p['n_area']:,p['n_area']:]
    Lambda_1=Lambda[0:p['n_area'],0:p['n_area']]
    # Lambda_4=Lambda[p['n_area']:,p['n_area']:]
    
    return Sigma_1, Lambda_1
    

    