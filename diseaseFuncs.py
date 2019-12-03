import numpy as np
import torch
import os
from sfw_torch import sfw_seis_torch, run_seis_information, run_seis_information_new

def load_data(disease):
    '''
    Loads .mat files provided by Sze as numpy matricies/arrays
    '''
    import scipy as sp
    import scipy.io
    import numpy as np
    if disease == 'tb':
        data = sp.io.loadmat('SIS_forBryan_TB.mat', struct_as_record=False, squeeze_me=True)
    elif disease == 'gon':
        data = sp.io.loadmat('SIS_forBryan_gon.mat', struct_as_record=False, squeeze_me=True)
    else:
        raise Exception('bad disease name: ' + disease)
    beta_array = data['beta']
    beta = []
    for i in range(beta_array.shape[0]):
        beta.append(np.matrix(beta_array[i]))
    f = data['f']
    p = data['p']
    n = data['n']
    G = np.matrix(data['G'])
    for j in range(len(f.I)):
        f.I[j] = np.matrix(f.I[j])
    return f, p, beta, n, G

def load_as_diag(fname, num_samples, n, oneminus = False):
    a = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=np.complex)
    a[np.isnan(a)] = 0
    a = np.real(a)
    a[np.isneginf(a)] = 0
    if oneminus:
        a = 1 - a
    mat = np.zeros((num_samples, n, n))
    for i in range(num_samples):
        mat[i] = np.diag(a[i])
    return torch.from_numpy(mat).double()

def load_as_diag_time(fname, num_samples, n, invert = False):
    a = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=np.complex)
    #yearly: take every 12th row
    a = a[::12]
    a[np.isnan(a)] = 0
    a = np.real(a)
    a[np.isneginf(a)] = 0
    a = torch.from_numpy(a).double()
    mat = torch.zeros(a.shape[0], num_samples, n, n).double()
    for i in range(a.shape[0]):
        if not invert:
            mat[i] = torch.diag(a[i]).expand_as(mat[i])
        else:
            inverse = 1./a[i]
            inverse[torch.isinf(inverse)] = 1
            mat[i] = torch.diag(inverse).expand_as(mat[i])
    return mat.double()


def load_as_vec(fname, num_samples, n):
    E = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=np.complex)
    E[np.isnan(E)] = 0
    E = np.real(E)
    E[np.isneginf(E)] = 0
    E = torch.from_numpy(E)
    E = E.view(num_samples, n, 1)
    return E.double()

def load_as_vec_time(fname, num_samples, n):
    E = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=np.complex)
    E = E[::12]
    E[np.isnan(E)] = 0
    E = np.real(E)
    E[np.isneginf(E)] = 0
    E = torch.from_numpy(E)
    E_expand = torch.zeros(E.shape[0], num_samples, n, 1)
    for i in range(E.shape[0]):
        E_expand[i] = E[i].unsqueeze(1).expand_as(E_expand[i])
    return E_expand.double()


def load_data_new(n, do_bootstrap=False, n_samples=1000):
    beta = np.loadtxt('annBeta.csv', delimiter=',', skiprows=1)
    beta_full = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            i_idx = min((int(i/5), beta.shape[0]-1))
            j_idx = min((int(j/5), beta.shape[0]-1))
            beta_full[i, j] = beta[i_idx, j_idx]
    beta = torch.from_numpy(beta_full)
    
    a = np.loadtxt('ann2018_alphaFastProb.csv.csv', delimiter=',', skiprows=1)
    num_samples = a.shape[0]
    
    alpha_fast = load_as_diag('ann2018_alphaFastProb.csv.csv', num_samples, n)
        
    alpha_slow = load_as_diag('ann2018_alphaSlowProb.csv.csv', num_samples, n)
   
    nu_sq = np.loadtxt('ann2018_clearanceProb.csv.csv', delimiter=',', skiprows=1)
    nu_sq[np.isnan(nu_sq)] = 0
    nu_sq = nu_sq.mean(axis = 0)
    nu_sq = torch.from_numpy(nu_sq)
    
    d = load_as_diag('ann2018_deathProb_TB.csv.csv', num_samples, n, oneminus=True)        
    
    mu = load_as_diag('ann2018_deathProb_nat.csv.csv', num_samples, n, oneminus=True)        
    
    S = load_as_diag_time('2018_hasHealthy.csv', num_samples, n)
    
    N = load_as_diag_time('2018_N.csv', num_samples, n, invert=True)
    
    E = load_as_vec_time('2018_haslatTB.csv', num_samples, n)
    
    I = load_as_vec_time('2018_hasTB.csv', num_samples, n)
    
    
    if do_bootstrap == True:
        S_boot = torch.zeros(S.shape[0], n_samples, *S.shape[2:]).double()
        for t in range(S.shape[0]):
            S_boot[t] = bootstrap(S[t], n_samples)
        
        E_boot = torch.zeros(E.shape[0], n_samples, *E.shape[2:]).double()
        for t in range(E.shape[0]):
            E_boot[t] = bootstrap(E[t], n_samples)
            
        I_boot = torch.zeros(I.shape[0], n_samples, *I.shape[2:]).double()
        for t in range(I.shape[0]):
            I_boot[t] = bootstrap(I[t], n_samples)

        N_boot = torch.zeros(N.shape[0], n_samples, *N.shape[2:]).double()
        for t in range(N.shape[0]):
            N_boot[t] = bootstrap(N[t], n_samples)
        
        beta = beta.expand_as(mu)
        alpha_fast = bootstrap(alpha_fast, n_samples)
        alpha_slow = bootstrap(alpha_slow, n_samples)
        mu = bootstrap(mu, n_samples)
        d = bootstrap(d, n_samples)
        beta = bootstrap(beta, n_samples)
        
        S = S_boot
        E = E_boot
        I = I_boot
        N = N_boot
        
    
    return S, E, I, N, alpha_fast, alpha_slow, beta, mu, d, nu_sq

def make_G(num_samples, n):
    G = torch.zeros(n, n)
    for i in range(n-1):
        G[i+1, i] = 1
    G = G.expand(num_samples, n, n)
    return G.double()
    

def bootstrap(A, n_samples):
    pop = list(range(A.shape[0]))
    indices = np.random.choice(pop, n_samples)
    indices = torch.tensor(indices).long()
    return A[indices]
            
def optimization(Thoriz, k, nu_max, betaScalar, betaInfo,betaInfoName, do_bootstrap,n_samples, num_itersVal):
    

    #nuBaseline is a vector
    #optTimeHorizon = 25
    #K = 0.3222222222222222222222
    K = 0.8333333333333
    disease = 'tb'
    
    n = 101
#    num_samples = 112
#    n_samples = 503
    
    #f, p, beta, n, G = load_data(disease)
    S, E, I, N, alpha_fast, alpha_slow, beta, mu, d, L = load_data_new(n, do_bootstrap, n_samples)
    print(S.type())
    G = make_G(n_samples, n)
    
    #U = np.ones((n)) * 0.05
    U = np.ones((n)) * 0.1
    U = U + L
    
    #totalTimeHorizon=31
    totalTimeHorizon=Thoriz
    
    #arbitrary constant value for nu_max
    nu_max = 0.5
    #nu_max = 0.05
    #0.3 original
    #U[U > nu_max] = nu_max 
    
    
    #use disease spread beta for information diffusion
    beta_full = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            i_idx = min((int(i/5), betaInfo.shape[0]-1))
            j_idx = min((int(j/5), betaInfo.shape[0]-1))
            beta_full[i, j] = betaInfo[i_idx, j_idx]
    beta_information = torch.from_numpy(beta_full)
    
    beta_information = betaScalar*beta_information
    #beta
    #10*beta 
    #0.00000001*beta
    #1000*beta
    #10000*beta
    
    
    #whether to load migration totals from file, or calculate from scratch
    load_cached_migration = False
    
    if load_cached_migration:
        migration_I = torch.load('migration_I.pt')
        migration_E = torch.load('migration_E.pt')
    else:
        #run the model with baseline clearance rate and no migration to get
        #the total migration
        migration_I = torch.zeros_like(I)
        migration_E = torch.zeros_like(E)
        for t in range(S.shape[0]):
            print('calibrating: {0}/{1}'.format(t, S.shape[0]))
            all_I, all_E, all_F = run_seis_information_new(t+1, G, S, I[0], migration_I, migration_E, L, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max)
            migration_I[t] = I[t] - all_I[t]
            migration_E[t] = E[t] - all_E[t]

    
    #sze debug

# =============================================================================
# 
#     if (k==1):
#         print('migration on')
#     
#     else:
#         print('turning off migraion')
#         migration_I = torch.zeros_like(I)
#         migration_E = torch.zeros_like(E)
# =============================================================================
    
    #code to run the model with a given nu (the argument "L" currently)
    all_I, all_E, all_F = run_seis_information_new(31, G, S, I[0], migration_I, migration_E, L, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max)

    dir_path = os.getcwd()
    os.chdir('0_currentOutput')    
    
    pathStr = 'nuMax'+str(nu_max)+'_betaMult'+str(betaScalar)+betaInfoName+'_t'+str(Thoriz)
    if not os.path.exists(pathStr):
        os.makedirs(pathStr)
        
    os.chdir(pathStr) 
    np.savetxt(str(k) +'SQ' + 'all_I.csv', all_I.squeeze().mean(dim=1).detach().numpy())
    np.savetxt(str(k) +'SQ' + 'all_E.csv', all_E.squeeze().mean(dim=1).detach().numpy())
    np.savetxt(str(k) +'SQ' + 'all_F.csv', all_F.squeeze().mean(dim=1).detach().numpy())
    #np.savetxt(str(k) +'SQ' + 'all_N.csv', all_N.squeeze().mean(dim=1).detach().numpy())
    np.savetxt(str(k) +'SQ' + 'nu_.csv', L.detach().numpy())
    os.chdir(dir_path)  
     
    #code to run the algorithm. Currently takes a while for the information spread model
    #Would probably be fine to reduce the number of iterations substantially
     
    
    import time
    start_time = time.time()

    nu_optimized = sfw_seis_torch(L, U, K, totalTimeHorizon, G, S, I[0], migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max, num_iters = num_itersVal)

    
    #1 iters
    #nu_optimized = sfw_seis_torch(L, U, K, totalTimeHorizon, G, S, I[0], migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max, num_iters=1)
    #nu_optimized = L
    
    #5 iters
    #nu_optimized = sfw_seis_torch(L, U, K, totalTimeHorizon, G, S, I[0], migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max, num_iters=5)

     
    #10 iters
    #nu_optimized = sfw_seis_torch(L, U, K, totalTimeHorizon, G, S, I[0], migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max, num_iters=30)
     
     
    #30 iters
    #nu_optimized = sfw_seis_torch(L, U, K, totalTimeHorizon, G, S, I[0], migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max, num_iters=30)
     
    #nu_optimized = sfw_seis_torch(L, U, K, totalTimeHorizon, G, S, I[0], migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max, num_iters=100)
     
    
    
    end_time = time.time()
    print("My program took", end_time-start_time, "to run")
    totTime = end_time-start_time
    
     
    #code to run the remodel with optimal nu 
    all_I, all_E, all_F = run_seis_information_new(31, G, S, I[0], migration_I, migration_E, nu_optimized, mu, d, beta, N, alpha_fast, alpha_slow, E[0], beta_information, nu_max)
     
     
    
    #save output as CSV
    #so that things are 2-d, this currently averages over the samples
    os.chdir('0_currentOutput')    
    os.chdir(pathStr)    
    np.savetxt(str(k) + 'all_I.csv', all_I.squeeze().mean(dim=1).detach().numpy())
    np.savetxt(str(k) + 'all_E.csv', all_E.squeeze().mean(dim=1).detach().numpy())
    np.savetxt(str(k) + 'all_F.csv', all_F.squeeze().mean(dim=1).detach().numpy())
    #np.savetxt(str(k) + 'all_N.csv', all_N.squeeze().mean(dim=1).detach().numpy())
    np.savetxt(str(k) + 'nu_optimized.csv', nu_optimized.detach().numpy())
    os.chdir(dir_path)    
    
#    return nu_optimized
    return totTime
 
def hilariousTest(num_samples, n, nu_max, beta_information):
    print('this is from hilarious Test')
    print('this is beta information')
    print(beta_information)
