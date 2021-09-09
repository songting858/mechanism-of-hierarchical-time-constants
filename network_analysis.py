# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:44:55 2020
Hierarchical Timescales in the Neocortex: Mathematical Mechanism and Biological Insights
@author: songting
"""
# %%
import pre_functions_clean as pf
import numpy as np

datafile='subgraph_data_macaque.pkl'

p = pf.load_data(datafile)

p,W = pf.genetate_net_connectivity(p)

#=======================pulse input to V1====================================
#pf.run_stimulus(p,VISUAL_INPUT=1,PULSE_INPUT=1)

#=======================white noise input to V1==============================
pf.plt_white_noise_input(p)

#==============eigenmode decomposition of the connectivity matrix============
#eigVecs_reorder, tau_reorder = pf.eig_decomposition(p,W)

#=======================perturbation analysis of the model===================
#pf.eigen_structure_approximation(p)     
