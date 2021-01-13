from __future__ import division

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import norm, invgamma
from scipy.special import logsumexp
#import matplotlib.pyplot as plt

#TODO: make all 3 random draw functions determinstic

# q_i, m_i, v_i^2
ksc_params = np.array([[0.04395, 2.77786,   0.16735],
                       [0.24566, 1.79518,   0.34023],
                       [0.34001, 0.61942,  0.64009],
                       [0.25750, -1.08819,  1.26261],
                       [0.10556, -3.97281,  2.61369],
                       [0.00002, -8.56686,  5.17950],
                       [0.00730, -10.12999, 5.79596]])


def scale_ksc_params(ksc_params, scale=1):
    ksc_params[:,1]*= scale
    ksc_params[:,2]*= scale**2
    return ksc_params    


# class QLSV(sm.tsa.statespace.MLEModel):
#     """
#     Quasi-likelihood stochastic volatility
#     """
#     def __init__(self, endog, offset=0.001):
#         # Convert to log squares
#         endog = np.log(endog**2 + offset)

#         # Initialize the base model
#         super(QLSV, self).__init__(endog, k_states=1, k_posdef=1,
#                                    initialization='stationary')

#         # Setup the observation covariance
#         self['obs_intercept', 0, 0] = -1.27
#         self['design', 0, 0] = 1
#         self['obs_cov', 0, 0] = np.pi**2 / 2
#         self['selection', 0, 0] = 1.

#     @property
#     def param_names(self):
#         return ['phi', 'sigma2_eta', 'mu']

#     @property
#     def start_params(self):
#         return np.r_[0.9, 100., 0.]

#     def transform_params(self, params):
#         return np.r_[constrain_stationary_univariate(params[:1]),
#                      params[1]**2, params[2:]]

#     def untransform_params(self, params):
#         return np.r_[unconstrain_stationary_univariate(params[:1]),
#                      params[1]**0.5, params[2:]]

#     def update(self, params, **kwargs):
#         super(QLSV, self).update(params, **kwargs)

#         gamma = params[2] * (1 - params[0])
#         self['state_intercept', 0, 0] = gamma
#         self['transition', 0, 0] = params[0]
#         self['state_cov', 0, 0] = params[1]


class TVLLDT(sm.tsa.statespace.MLEModel):
    """
    Time-varying local linear deterministic trend
    """
    def __init__(self, endog, offset=0.00001):
        # Convert to log squares, with offset
        endog = np.log(endog**2 + offset)

        # Initialize base model
        super(TVLLDT, self).__init__(endog, k_states=1, k_posdef=1,
                                     initialization='stationary')

        # Setup time-varying arrays for observation equation
        self['obs_intercept'] = np.zeros((1, self.nobs))
        self['obs_cov'] = np.zeros((1, 1, self.nobs))

        # Setup fixed components of state space matrices
        self['design', 0, 0] = 1
        self['selection', 0, 0] = 1

    def update_mixing(self, indicators):
        # z_t | s_t ~ N(m_i - 1.27036, v_i^2)
        # See equation (10), p. 371
        self['obs_intercept', 0] = ksc_params[indicators, 1] - 1.27036
        self['obs_cov', 0, 0] = ksc_params[indicators, 2]

    def update(self, params, **kwargs):
        params = super(TVLLDT, self).update(params, **kwargs)

        self['state_intercept', 0, 0] = params[0] * (1 - params[1])
        self['transition', 0, 0] = params[1]
        self['state_cov', 0, 0] = params[2]


def mixing_posterior_t(y, h):
    K = ksc_params.shape[0]

    log_posterior_kernel = np.zeros(K)
    posterior_kernel = np.zeros(K)
    for i in range(K):
        prior = ksc_params[i, 0]
        mean = h + ksc_params[i, 1] - 1.27036
        resid = y - mean
        print(i, resid)
        variance = ksc_params[i, 2]
        # print np.r_[i, prior, y, h, ksc_params[i, 1], mean, resid, variance]

        llf = -0.5 * (np.log(2 * np.pi * variance) + (resid**2) / variance)
        # print(i, llf)
        log_posterior_kernel[i] = np.log(prior) + llf

        lf = np.exp(llf)
        posterior_kernel[i] = prior * lf

    # print('a', posterior_kernel / np.sum(posterior_kernel))
    posterior_kernel = np.exp(log_posterior_kernel)
    # print('b', posterior_kernel / np.sum(posterior_kernel))
    posterior = posterior_kernel / np.sum(posterior_kernel)

    return posterior


def draw_mixing_t(y, h):
    posterior = mixing_posterior_t(y, h)

    variate = np.random.uniform()
    tmp = np.cumsum(posterior) > variate
    sample = np.argmax(tmp)

    return sample


def draw_mixing_new(mod, states):
    y = mod.endog[:, 0]
    h = states[0]    

    mixing = np.zeros(mod.nobs)
    for t in range(mod.nobs):
        mixing[t] = draw_mixing_t(y[t], h[t])

    return mixing


def mixing_posterior(mod, states):
    resid = mod.endog[:, 0] - states[0]

    # Construct the means (nobs x 7), variances (7,), prior probabilities (7,)
    means = ksc_params[None, :, 1] - 1.27036
    variances = ksc_params[:, 2]
    prior_probabilities = ksc_params[:, 0]

    # Make dimensions compatible for broadcasting
    resid = np.repeat(resid[:, None], len(variances), axis=-1)
    variances = np.repeat(variances[None, :], mod.nobs, axis=0)
    prior_probabilities = np.repeat(prior_probabilities[None, :], mod.nobs,
                                    axis=0)

    # Compute loglikelihood (nobs x 7)
    loglikelihoods = -0.5 * ((resid - means)**2 / variances +
                             np.log(2 * np.pi * variances))

    # Get (values proportional to) the (log of the) posterior (nobs x 7)
    posterior_kernel = loglikelihoods + np.log(prior_probabilities)

    # Normalize to get the actual posterior probabilities
    tmp = logsumexp(posterior_kernel, axis=1)
    posterior_probabilities = np.exp(posterior_kernel - tmp[:, None])

    return posterior_probabilities

def draw_mixing(mod, states):
    posterior_probabilities = mixing_posterior(mod, states)

    # Draw from the posterior
    variates = np.random.uniform(size=mod.nobs)
    tmp = np.cumsum(posterior_probabilities, axis=1) > variates[:, None]
    sample = np.argmax(tmp, axis=1)

    return sample


def g(phi, states, mu, sigma2, prior_params=(20, 1.5)):
    phi_1, phi_2 = prior_params

    # Prior distribution gives zero weight to non-stationary processes
    if np.abs(phi) >= 1:
        return -np.inf

    prior = ((1 + phi) / 2)**(phi_1 - 1) * ((1 - phi) / 2)**(phi_2 - 1)

    tmp1 = (states[0, 0] - mu)**2 * (1 - phi**2) / 2 * sigma2
    tmp2 = 0.5 * np.log(1 - phi**2)

    return np.log(prior) - tmp1 + tmp2


def draw_posterior_phi(model, states, phi, mu, sigma2, prior_params=(20, 1.5)):
    tmp1 = np.sum((states[0, 1:] - mu) * (states[0, :-1] - mu))
    tmp2 = np.sum((states[0, :-1] - mu)**2)
    phi_hat = tmp1 / tmp2
    V_phi = sigma2 / tmp2

    proposal = norm.rvs(phi_hat, scale=V_phi**0.5)
    g_proposal = g(proposal, states, mu, sigma2, prior_params)
    g_previous = g(phi, states, mu, sigma2, prior_params)
    acceptance_probability = np.exp(g_proposal - g_previous)

    return proposal if acceptance_probability > np.random.uniform() else phi


def draw_posterior_sigma2(model, states, phi, mu, prior_params=(5, 0.05)):
    sigma_r, S_sigma = prior_params

    v1 = sigma_r + model.nobs
    tmp1 = (states[0, 0] - mu)**2 * (1 - phi**2)
    tmp1 = 0
    tmp = np.sum(((states[0, 1:] - mu) - phi * (states[0, :-1] - mu))**2)
    
    delta1 = S_sigma + tmp1 + tmp

    #print('draw_posterior_sigma2', delta1/v1)

    return invgamma.rvs(v1, scale=delta1)


def draw_posterior_mu(model, states, phi, sigma2):
    sigma2_mu = sigma2 / ((model.nobs - 1) * (1 - phi)**2 + (1 - phi**2))

    tmp1 = ((1 - phi**2) / sigma2) * states[0, 0]
    tmp = ((1 - phi) / sigma2) * np.sum(states[0, 1:] - phi * states[0, :-1])
    mu_hat = sigma2_mu * (tmp1 + tmp)

    return norm.rvs(loc=mu_hat, scale=sigma2_mu**0.5)

    

def estimate_stoch_vol(endog, random_state, params=(100,50,1,None), initial_values=(0,0,0.95,0.5), ksc_scale=1, plot=False):    
    """
    Parameters
    ----------
    endog : TYPE
        DESCRIPTION.
    random_state : TYPE
        e.g. random_state = np.random.RandomState(10)
    params : TYPE, optional
        (n_iterations, burn, thin, trace selected as simulated output(when it is None, choose a random one after burned traces)). The default is (100,50,1,None).
    initial_values : TYPE, optional
        (selection of ksc distribution, mu(state mean), phi(state auto-regressive coefficient), sigma2(state variance)). The default is (0,0,0.95,0.5).
    ksc_scale : TYPE, optional
        how much to scale the variance of observation innovation (follow ksc distribution). The default is 1.
    plot : boolean, optional
        whether to plot analysis or not. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # Simulation parameters
    n_iterations = params[0]
    burn = params[1]
    thin = params[2]
    trace_selected = params[3]
    
    # q_i, m_i, v_i^2
    ksc_params = np.array([[0.04395, 2.77786,   0.16735],
                           [0.24566, 1.79518,   0.34023],
                           [0.34001, 0.61942,  0.64009],
                           [0.25750, -1.08819,  1.26261],
                           [0.10556, -3.97281,  2.61369],
                           [0.00002, -8.56686,  5.17950],
                           [0.00730, -10.12999, 5.79596]])

    ksc_params = scale_ksc_params(ksc_params, ksc_scale)   

            
    # Setup the model and simulation smoother
    mod = TVLLDT(endog)
    mod.set_smoother_output(0, smoother_state=True)
    sim = mod.simulation_smoother()

    # Storage for traces
    #trace_smoothed = np.zeros((n_iterations + 1, mod.nobs))
    trace_states = np.zeros((n_iterations + 1, mod.nobs))
    trace_mixing = np.zeros((n_iterations + 1, mod.nobs), int)
    trace_mu = np.zeros((n_iterations + 1, 1))
    trace_phi = np.zeros((n_iterations + 1, 1))
    trace_sigma2 = np.zeros((n_iterations + 1, 1))
    
    # Initial values (p. 367)
    trace_mixing[0] = initial_values[0]
    trace_mu[0] = initial_values[1]
    trace_phi[0] = initial_values[2]
    trace_sigma2[0] = initial_values[3]

    
    # Iterations
    for s in range(1, n_iterations + 1):
        # Update the parameters of the model
        mod.update_mixing(trace_mixing[s-1])
        params = np.r_[trace_mu[s-1], trace_phi[s-1], trace_sigma2[s-1]]
        mod.update(params, transformed=True)
    
        # Simulation smoothing
        sim.simulate()
        states = sim.simulated_state
        trace_states[s] = states[0]
        #trace_intercept[s] = mod['obs_intercept']
    
        # Draw mixing indicators
        trace_mixing[s] = draw_mixing(mod, states)
    
        # Draw parameters
        trace_phi[s] = draw_posterior_phi(mod, states, trace_phi[s-1], trace_mu[s-1], trace_sigma2[s-1])
        trace_sigma2[s] = draw_posterior_sigma2(mod, states, trace_phi[s-1], trace_mu[s-1])
        #trace_sigma2[s] = 10000
        trace_mu[s] = draw_posterior_mu(mod, states, trace_phi[s-1], trace_sigma2[s-1])
        
    if plot:
        posterior_states = np.percentile(trace_states[burn::thin], (15, 50, 85), axis=0)         
        pd.DataFrame((np.exp(posterior_states[1]/2)),columns=['predicted_std']).join(pd.Series(endog,name='endog')).plot()
        pd.DataFrame((np.exp(trace_states[200]/2)),columns=['simulated_std']).join(pd.Series(endog,name='endog')).plot()
        
        #pd.DataFrame((np.exp(posterior_intercept[1]/2)),columns=['predicted_intercept']).join(pd.Series(np.log(endog**2),name='actual_endog')).plot()
        
        pd.DataFrame(mod.endog, columns=['actual_endog']).join(pd.Series(posterior_states[1],name='posterior_state')).plot()
        pd.DataFrame(mod.endog, columns=['actual_endog']).join(pd.Series(trace_states[200],name='sample_state')).plot()
        
        #pd.DataFrame((np.exp(0.5*trace_states[20]))).join(pd.Series(endog,name='endog')).plot()
        pd.DataFrame(trace_sigma2).iloc[50:].plot()
        #pd.DataFrame(trace_phi).plot()
    
    if pd.isnull(trace_selected):
        trace_selected = int(random_state.uniform(burn, n_iterations))
    
    simulated_states = trace_states[trace_selected]
    
    return np.exp(simulated_states/2), (trace_mixing[trace_selected], trace_mu[trace_selected][0], trace_phi[trace_selected][0], trace_sigma2[trace_selected][0])
    
    

if __name__ == '__main__':
    ### test usage
    # sample endog
    np.random.seed(1234)
    endog = pd.Series(list(np.random.normal(0,0.1,100)) + list(np.random.normal(0,0.5,20))+list(np.random.normal(0,0.15,100))) 
    #endog = pd.Series(list(np.random.normal(0,1,100)) + list(np.random.normal(0,5,20))+list(np.random.normal(0,1.5,100))) 
    
    #endog*=10000000
    #endog = endog *2
    # Setup the model and simulation smoother
    mod = TVLLDT(endog)
    mod.set_smoother_output(0, smoother_state=True)
    sim = mod.simulation_smoother()
    
    # Simulation parameters
    n_iterations = 1000
    burn = 50
    thin = 1
    
    # Storage for traces
    #trace_smoothed = np.zeros((n_iterations + 1, mod.nobs))
    trace_states = np.zeros((n_iterations + 1, mod.nobs))
    trace_mixing = np.zeros((n_iterations + 1, mod.nobs), int)
    trace_mu = np.zeros((n_iterations + 1, 1))
    trace_phi = np.zeros((n_iterations + 1, 1))
    trace_sigma2 = np.zeros((n_iterations + 1, 1))
    #trace_intercept = np.zeros((n_iterations + 1, mod.nobs))
    
    # Initial values (p. 367)
    trace_mixing[0] = 0
    trace_mu[0] = 0.0
    trace_phi[0] = 0.99
    trace_sigma2[0] = 10000
    # Iterations
    for s in range(1, n_iterations + 1):
        # Update the parameters of the model
        mod.update_mixing(trace_mixing[s-1])
        params = np.r_[trace_mu[s-1], trace_phi[s-1], trace_sigma2[s-1]]
        mod.update(params, transformed=True)
    
        # Simulation smoothing
        sim.simulate()
        states = sim.simulated_state
        trace_states[s] = states[0]
        #trace_intercept[s] = mod['obs_intercept']
    
        # Draw mixing indicators
        trace_mixing[s] = draw_mixing(mod, states)
    
        # Draw parameters
        trace_phi[s] = draw_posterior_phi(mod, states, trace_phi[s-1], trace_mu[s-1], trace_sigma2[s-1])
        trace_sigma2[s] = draw_posterior_sigma2(mod, states, trace_phi[s-1], trace_mu[s-1])
        #trace_sigma2[s] = 10000
        trace_mu[s] = draw_posterior_mu(mod, states, trace_phi[s-1], trace_sigma2[s-1])
        
    posterior_states = np.percentile(trace_states[burn::thin], (15, 50, 85), axis=0)
    #posterior_intercept = np.percentile(trace_intercept[burn::thin], (15, 50, 85), axis=0)
    
    rs_state = posterior_states[1]
    
    
    #pd.DataFrame(posterior_states[1]).join(pd.Series(endog_y,name='endog_y')).plot()
     
    pd.DataFrame((np.exp(posterior_states[1]/2)),columns=['predicted_std']).join(pd.Series(endog,name='endog')).plot()
    pd.DataFrame((np.exp(trace_states[200]/2)),columns=['simulated_std']).join(pd.Series(endog,name='endog')).plot()
    
    #pd.DataFrame((np.exp(posterior_intercept[1]/2)),columns=['predicted_intercept']).join(pd.Series(np.log(endog**2),name='actual_endog')).plot()
    
    pd.DataFrame(mod.endog, columns=['actual_endog']).join(pd.Series(posterior_states[1],name='posterior_state')).plot()
    pd.DataFrame(mod.endog, columns=['actual_endog']).join(pd.Series(trace_states[200],name='sample_state')).plot()
    
    #pd.DataFrame((np.exp(0.5*trace_states[20]))).join(pd.Series(endog,name='endog')).plot()
    pd.DataFrame(trace_sigma2).iloc[50:].plot()
    #pd.DataFrame(trace_phi).plot()
