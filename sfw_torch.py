import torch
import numpy as np

def run_sis(T, G, S, I, newI, nu, mu, d, beta, N):
    '''
    Runs the linearized SIS model, returning the total number of infected agents
    summed over all time steps.
    '''
    #duplicate these variables along an additional axis to match the batch size
    nu = torch.diag(1 - nu).expand_as(beta)
    d = torch.diag(1 - d).expand_as(beta)
    G = G.expand_as(beta)
    #run the main loop for the linearized disease dynamics
    total_infected = I.mean(dim=0).sum()
    for t in range(1, T):
        new_infections = S[t-1] @ mu[t-1] @ beta @ N[t-1]
        old_infections = nu @ d
        A = G @ (new_infections + old_infections)
        I = A @ I
        I = I + newI[t]
        total_infected += I.mean(dim=0).sum() 
    return total_infected


def run_seis(T, G, S, I, newI, nu, mu, d, beta, N, alpha_fast, alpha_slow, E):
    '''
    Runs the linearized SEIS model, returning the total number of infected agents
    summed over all time steps.
    '''
    #duplicate these variables along an additional axis to match the batch size
    nu = torch.diag(1 - nu).expand_as(beta)
    d = torch.diag(1 - d).expand_as(beta)
    G = G.expand_as(beta)
    E = E.expand_as(beta)
    alpha_fast = torch.diag(alpha_fast).expand_as(beta)
    alpha_slow = alpha_slow.expand_as(beta)
    #run the main loop for the linearized disease dynamics
    total_infected = I.mean(dim=0).sum()
    import numpy as np
    infections_time = np.zeros((T, 100))
    for t in range(1, T):
        new_infections = S[t-1] @ mu[t-1] @ beta @ N[t-1] @ I
#        print(new_infections.shape)
#        print(alpha_fast.shape)
        new_infections_active = alpha_fast @ new_infections
        new_infections_latent = new_infections - new_infections_active
        E = mu[t-1] @ E
        activations = alpha_slow*E
        E = (1 - alpha_slow)*E
#        print(E[:, 0].shape)
#        print(new_infections_latent.squeeze().shape)
        E[:, 0] += new_infections_latent.squeeze()
        E = G @ E @ G
        old_infections = nu @ d @ I
#        print(new_infections_active.shape)
#        print(old_infections.shape)
#        print(newI[t].shape)
#        print(activations.sum(dim=2).shape)
#        print(I.shape)
        I = new_infections_active + old_infections + newI[t] + activations.sum(dim=2).view_as(I)
        I = G @ I
#        infections_time.append(I.mean(dim=0).sum().item())
        for j in range(100):
            total_pop = (1./torch.diag(N[t, j])).sum()
            infections_time[t, j] = I[j].sum()/total_pop
            if t == 1 and j == 0:
                print(total_pop.item(), I[j].sum().item())
#        total_infected += I.mean(dim=0).sum() 
    return infections_time


def vector_to_diag(not_informed_fraction, beta):
    not_informed_fraction_diag = torch.zeros_like(beta)
    for s in range(beta.shape[0]):
        not_informed_fraction_diag[s] = torch.diag(not_informed_fraction[s].squeeze())
#    print('diag output')
#    print(not_informed_fraction_diag)
    return not_informed_fraction_diag


def run_seis_information(T, G, S, I, newI, nu, mu, d, beta, N, alpha_fast, alpha_slow, E, beta_information, nu_max):
    '''
    Runs the linearized SEIS model, returning the total number of infected agents
    summed over all time steps.
    '''
    #duplicate these variables along an additional axis to match the batch size
    informed = nu.view(len(nu), 1)
    informed = informed.expand(beta.shape[0], *informed.shape)
    nu = torch.diag(1 - nu).expand_as(beta)
    d = torch.diag(1 - d).expand_as(beta)
    G = G.expand_as(beta)
    E = E.expand_as(beta)
    alpha_fast = torch.diag(alpha_fast).expand_as(beta)
    alpha_slow = alpha_slow.expand_as(beta)
    
    #keep track of infected, latent, and informed at each time step
    all_infections = torch.zeros(T, beta.shape[1], 1)
    all_E = torch.zeros(T, E.shape[1], E.shape[2])
    all_F = torch.zeros_like(all_infections)
    
    #run the main loop for the linearized disease dynamics
    for t in range(1, T):
        #update nu with new information spread
        not_informed_fraction = 1 - informed
        not_informed_fraction_diag = vector_to_diag(not_informed_fraction, beta)
        #constant scaling the beta for information spread
        informed = 0.1*not_informed_fraction_diag@beta_information@informed + informed
        nu = nu_max*informed
        nu = vector_to_diag(1 - nu, beta)
        
        #infections
        new_infections = S[t-1] @ mu[t-1] @ beta @ N[t-1] @ I
        new_infections_active = alpha_fast @ new_infections
        new_infections_latent = new_infections - new_infections_active
        E = mu[t-1] @ E
        activations = alpha_slow*E
        E = (1 - alpha_slow)*E
        E[:, 0] += new_infections_latent.squeeze()
        E = G @ E @ G
        old_infections = nu @ d @ I
        I = new_infections_active + old_infections + newI[t] + activations.sum(dim=2).view_as(I)
        I = G @ I
        
        #return E, I, F by time and age group
        #mean across samples
        all_infections[t] = I.mean(dim=0)
        all_E[t] = E.mean(dim=0)
        all_F[t] = informed.mean(dim = 0)
        
    return all_infections, all_E, all_F


def run_seis_information_new(T, G, S, I, migration_I, migration_E, nu, mu, d, beta, N, alpha_fast, alpha_slow, E, beta_information, nu_max):
    '''
    Runs the linearized SEIS model, returning the total number of infected agents
    summed over all time steps.
    '''
    #read in for first period of F, informed
    #nu_sq = np.loadtxt('ann2018_clearanceProb.csv.csv', delimiter=',', skiprows=1)
    #nu_sq[np.isnan(nu_sq)] = 0
    #nu_sq = nu_sq.mean(axis = 0)
    #nu_sq = torch.from_numpy(nu_sq)

    #duplicate these variables along an additional axis to match the batch size
    beta = beta.expand_as(G)
    informed = nu.view(len(nu), 1)
    informed = informed.expand(beta.shape[0], *informed.shape)
    nu = torch.diag(1 - nu).expand_as(beta)
    num_samples = G.shape[0]
    #keep track of infected, latent, and informed at each time step
    all_I = torch.zeros(T, num_samples, beta.shape[1], 1).double()
    all_E = torch.zeros(T, num_samples, E.shape[1], E.shape[2]).double()
    all_F = torch.zeros_like(all_I).double()
    all_I[0] = I[0]
    all_E[0] = E[0]
    #all_I[0] = I[30]
    #all_E[0] = E[30]

    all_F[0] = informed
    
    #run the main loop for the linearized disease dynamics
    for t in range(1, T):
        #update nu with new information spread
        not_informed_fraction = 1 - informed
        not_informed_fraction_diag = vector_to_diag(not_informed_fraction, beta)
        #constant scaling the beta for information spread
        informed = not_informed_fraction_diag@beta_information@informed + informed
        #print('here is info beta mat')
        #print(beta_information)
        #print('here is informed')
        #print(informed)  
        #debug sze
        nu = nu_max*informed
        nu = vector_to_diag(1 - nu, beta)
        
        #infections
        new_infections = S[t-1] @ mu @ beta @ N[t-1] @ I
        new_infections_active = alpha_fast @ new_infections
        new_infections_latent = new_infections - new_infections_active
        E = mu @ E
        activations = alpha_slow@E
        E = E - activations
        E += new_infections_latent
        E = G @ E + migration_E[t] #CHANGING TO USING THE LAST MIGRATION PERIOD
        #E = G @ E + migration_E[30]
        
        
        old_infections = nu @ d @ I
        I = new_infections_active + old_infections + activations
        I = G @ I + migration_I[t]   #CHANGING TO USING THE LAST MIGRATION PERIOD
        #I = G @ I + migration_I[30]

        
        #return E, I, F by time and age group
        #mean across samples
        all_I[t] = I
        all_E[t] = E
        all_F[t] = informed
        
    #print(all_I)
    return all_I, all_E, all_F
    

class SISInstance():
    """
    Represents an instantiation of the SIS model with a particular (distribution 
    over) parameters. Foward pass computes total infections as a function of nu,
    backward computes gradient wrt nu.
    """
    def __init__(self, T, G, S, I, newI, mu, d, beta, N):
        self.T = T
        self.G = G
        self.S = S
        self.I = I
        self.newI = newI
        self.mu = mu
        self.d = d
        self.beta = beta
        self.N = N
    
    def __call__(self, nu):
        return run_sis(self.T, self.G, self.S, self.I, self.newI, nu, self.mu, self.d, self.beta, self.N)
    

def greedy(grad, U, L, K):
    '''
    Greedily select budget number of elements with highest weight according to
    grad
    '''
    sorted_groups = torch.sort(grad)[1]
    nu = L.clone()
    curr = 0
    while (nu - L).sum() < K and curr < len(grad):
        amount_add = min([U[sorted_groups[curr]] - L[sorted_groups[curr]], K - (nu - L).sum()])
        nu[[sorted_groups[curr]]] += amount_add
        curr += 1
    return nu

def sfw_torch(L, U, K, T, G, S, I, newI, mu, d, beta, N, num_iters = 100):
    sis = SISInstance(T, G, S, I, newI, mu, d, beta, N)
    nu = torch.rand_like(L, requires_grad=True)
    nu.data.zero_()
    nu.grad = torch.zeros_like(nu)
    for i in range(num_iters):
        val = sis(nu + L)
        nu.grad.zero_()
        val.backward()
        nu.data += 1./num_iters * greedy(nu.grad, U - L, torch.zeros_like(nu), K)
    nu.data += L
    return nu



def sfw_seis_torch(L, U, K, T, G, S, I, migration_I, migration_E, mu, d, beta, N, alpha_fast, alpha_slow, E, beta_information, nu_max, num_iters = 100):
    nu = torch.rand_like(L, requires_grad=True)
    nu.data.zero_()
    nu.grad = torch.zeros_like(nu)
    for i in range(num_iters):
        print('optimizing: {}/{}'.format(i, num_iters))
        all_I, all_E, all_F = run_seis_information_new(T, G, S, I, migration_I, migration_E, nu + L, mu, d, beta, N, alpha_fast, alpha_slow, E, beta_information, nu_max)
        val = all_I.sum()
        nu.grad.zero_()
        val.backward()
        nu.data += 1./num_iters * greedy(nu.grad, U - L, torch.zeros_like(nu), K)
    nu.data += L
    #print(nu)
    return nu

        
    