# -*- coding: utf-8 -*-
"""
Created on Nov 10 2019

# Takes in upser inputs and runs the analysis
"""

import os

import numpy as np
import torch
from sfw_torch import sfw_seis_torch, run_seis_information, run_seis_information_new
from diseaseFuncs import load_data, load_as_diag, load_as_diag_time, load_as_vec, load_as_vec_time, load_data_new, make_G, optimization, hilariousTest
 


# =============================================================================
#       User Inputs
# =============================================================================

# Time Horizon
Thoriz       = 10

# Max Clearance
nu_max = 0.5

# beta matrix: which mixing pattern?
betaScalar   = 5000
betaInfoName = 'ann'
betaInfo = np.loadtxt('annBeta.csv', delimiter=',', skiprows=1)

# Optimization Preferences
do_bootstrap = True
n_samples    = 50
num_itersVal = 10



# =============================================================================
#       Run Analysis
# =============================================================================

    
totTime = optimization(Thoriz, k, nu_max, betaScalar, betaInfo,betaInfoName, do_bootstrap, n_samples, num_itersVal)

