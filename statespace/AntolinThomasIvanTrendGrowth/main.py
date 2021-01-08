##################### Example ####################################
from __future__ import division

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.tools import (
    constrain_stationary_univariate, unconstrain_stationary_univariate)

from scipy.stats import norm, multivariate_normal, invgamma
from scipy.special import logsumexp
from scipy.signal import lfilter

OBS_INITIAL_AR_BURN = 2
RANDOM_STATE =  np.random.RandomState(11) # deterministic random data

def draw_posterior_sigma2(X, Y, beta=np.array([1]), prior_params=(1, 0.001)):
    resid = Y - X.dot(beta) 
    post_shape = prior_params[0] + len(resid) + 1
    post_scale = prior_params[1] + np.sum(resid**2)
    
    # return invgamma(shape, scale), shape: n_obs, or degree of freedom; scale: sum of variance variable
    # scale/shape decides expected value of the variance variable
    # scale is sample size, it decides the dispersion of the variance variable
    # return invgamma(post_shape, scale=post_scale).rvs()
    return 1/RANDOM_STATE.gamma(post_shape,1/post_scale)


def draw_beta_gls(X, Y, error_cov, reject_explosive=False):
    """
    http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/4-6-Multiple-GLS.html
    """    
    nobs = len(Y)
    error_cov_inv = np.linalg.inv(error_cov)

    beta_hat = np.linalg.inv(X.T.dot(error_cov_inv).dot(X)) \
        .dot(X.T.dot(error_cov_inv).dot(Y))
    
    sigma2_hat = (Y-X.dot(beta_hat)).T \
                    .dot(error_cov_inv) \
                    .dot(Y-X.dot(beta_hat)) \
                    /(nobs-2-1)
                    
    beta_var = sigma2_hat * np.linalg.inv(X.T.dot(error_cov_inv).dot(X))
    
    beta_drawed = RANDOM_STATE.multivariate_normal(mean=beta_hat,cov=beta_var)
    if reject_explosive:
        while(abs(sum(beta_drawed)) > 1):
            print('draw_posterior_beta_gls, reject a draw of {} because it is explosive'.format(beta_drawed))
            beta_drawed = RANDOM_STATE.multivariate_normal(mean=beta_hat,cov=beta_var)    
        
    return beta_drawed



class ATITG(sm.tsa.statespace.MLEModel):
    """
    Paper: Following the Trend: Tracking GDP when Long-Run Growth is Uncertain.
    by Juan Antolin-Diaz, Thomas Drechsel, Ivan Petrella
    """
    def __init__(self, endog, phi, stoch_vol, k_states=15,offset=0.001):
        """
        Parameters
        ----------
        endog : pd.DataFrame
            DESCRIPTION.
        phi : dict 
            system parameters
        stoch_vol : dict
            stochastic volitility 
        k_states : int, optional
            # of states. The default is 15.
        offset : float, optional
            TODO. The default is 0.001.

        Returns
        -------
        None.

        """
        
        # Initialize the base model
        super(ATITG, self).__init__(endog, 
                                    k_states=k_states, # dimension of states
                                    k_posdef=k_states  # dimension of state disturbances
                                )
        self.initialize_approximate_diffuse(variance=10**4) # X0 ∼ N(0, 104)
        
        self['design'] = self.get_design_matrix(phi)
        self['transition'] = self.get_transition_matrix(phi)
        self['selection'] = np.eye(k_states)
        self['obs_cov'] = self.get_obs_cov(stoch_vol['stoch_vol_obs_innov'])
        self['state_cov'] = self.get_state_cov(phi['param_var_trend'], stoch_vol, k_states) 
        

    
    def get_design_matrix(self, phi):
        H = [[1/3,2/3,1,2/3,1/3]*3]
        for design_beta, ar_betas in zip(phi['param_cycle_design_betas'], phi['param_innov_ar_betas']):
            params = [design_beta, -design_beta*ar_betas[0], -design_beta*ar_betas[1]]
            H.append([0]*5 + params + [0]*7)
        
        return np.array(H)
    
    
    def get_transition_matrix(self, phi):
        F1 = np.concatenate([
                np.array([[1,0,0,0,0]]),
                np.concatenate([
                    np.eye(4),
                    np.zeros([4,1])
                ],axis=1)
            ])
        F2 = np.concatenate([
                np.array([phi['param_cycle_ar_betas']+[0,0,0]]),
                np.concatenate([
                    np.eye(4),
                    np.zeros([4,1])
                ],axis=1)        
            ])
        F3 = np.concatenate([
                np.array([phi['param_innov1_ar_betas']+[0,0,0]]),
                np.concatenate([
                    np.eye(4),
                    np.zeros([4,1])
                ],axis=1)        
            ])
        
        F = np.concatenate([
                np.concatenate([
                    F1,
                    np.zeros([5,5]),
                    np.zeros([5,5])
                ],axis=1),
                np.concatenate([
                    np.zeros([5,5]),
                    F2,
                    np.zeros([5,5])
                ],axis=1),
                np.concatenate([
                    np.zeros([5,5]),
                    np.zeros([5,5]),
                    F3
                ],axis=1)
            ])
    
        return F
    
    
    def get_obs_cov(self, obs_innovs_list):
        n = len(obs_innovs_list) + 1
        m = len(obs_innovs_list[0])
        print('get_obs_cov, dimention: n: {}, m: {}'.format(n,m))
        nums_all = [0] * m * n 
        for i in range(len(obs_innovs_list)):
            nums_all+=[0]*m
            nums_all+= [0]*m*i
            obs_innovs = obs_innovs_list[i]
            nums_all+= obs_innovs 
            nums_all+= [0]*m*(n-1-i-1)
    
        R = np.reshape(nums_all, (n,n,m))
        
        return R
    
    
    def get_state_cov(self, param_var_trend, stoch_vol, k_states):
        vols_cycle = stoch_vol['stoch_vol_cycle']
        vols_innov = stoch_vol['stoch_vol_innov']
     
        m = len(vols_cycle)
        nums_all = []
        
        # construct a one-dimensional list first, then reshape to 3-dimensional state covariance matrix
        for row_id in range(k_states):
            for col_id in range(k_states):
                if (row_id==0) and (col_id==0):
                    nums_all+= [param_var_trend]*m
                elif (row_id==5) and (col_id==5):
                    nums_all+= vols_cycle
                elif (row_id==10) and (col_id==10):
                    nums_all+= vols_innov
                else:
                    nums_all+= [0]*m
                    
        Q = np.reshape(nums_all, (k_states,k_states,m))
        
        return Q
    


    @property
    def param_names(self):
        return []

    @property
    def start_params(self):
        return []

    # def transform_params(self, params):
    #     return np.r_[constrain_stationary_univariate(params[:1]),
    #                  params[1]**2, params[2:]]

    # def untransform_params(self, params):
    #     return np.r_[unconstrain_stationary_univariate(params[:1]),
    #                  params[1]**0.5, params[2:]]

    def update(self, params, **kwargs):
        if ('phi' in kwargs) and ('stoch_vol' in kwargs): 
            k_states, phi, stoch_vol = self.k_states, kwargs['phi'], kwargs['stoch_vol']
            kwargs = {key:kwargs[key] for key in kwargs if key not in ['phi','stoch_vol']}      
            self['design'] = self.get_design_matrix(phi)
            self['transition'] = self.get_transition_matrix(phi)
            self['selection'] = np.eye(k_states)
            self['obs_cov'] = self.get_obs_cov(stoch_vol['stoch_vol_obs_innov'])
            self['state_cov'] = self.get_state_cov(phi['param_var_trend'], stoch_vol, k_states) 

        super(ATITG, self).update(params, **kwargs)
        


####################################################################################################        

########
## Loading Data
# expect: first col is official Quarterly GDP; remainly columns are detrended monthly indicators
RNG = np.random.RandomState(10) # deterministic random data

def get_dummy_us_growth_endog(n=30, obs_innov_params=(0.5, 5), trend_mav_periods=120, cycle_mav_periods=9):
    raw_data = pd.read_csv(r'local/raw_us_gdp.csv',index_col=0).set_index('DATE')
    raw_data.index = raw_data.index.map(pd.Timestamp)
    raw_data = raw_data.resample('Q').last()
    data = (raw_data/raw_data.shift(4)-1).resample('M').ffill()
    data.columns = ['y']
    data.loc[:, 'trend'] = data['y'].rolling(trend_mav_periods).mean()
    data.loc[:, 'cycle_n_innov'] = data['y'] - data['trend']
    data.loc[:, 'cycle'] = data['cycle_n_innov'].rolling(cycle_mav_periods).mean()
    data = data.loc['2000':]
    m = len(data)
    cycle_std = data['cycle'].std()
    for i in range(n):
        chunk = (i//obs_innov_params[1] + 1) * obs_innov_params[0]
        rand_nums = RNG.normal(size=m)
        obs_innov = rand_nums * cycle_std * chunk
        data['y{}'.format(i)] = data['cycle'] + obs_innov
    
    data_observation = data[[col for col in data.columns if (col[0]=='y') and (len(col)<4)]]
    
    # change y from monthly series to quarterly series -> to match with quarterly released growth rate.
    data_observation.loc[:, 'dummy_yq'] = data_observation.index.map(lambda dt: "{}-{}".format(dt.year, dt.quarter))
    data_observation.loc[:, 'dummy_keep'] = ~data_observation['dummy_yq'].duplicated(keep='last')
    y_q = data_observation.loc[data_observation['dummy_keep'],['y']]
    data_observation = data_observation[[col for col in data_observation.columns if 'dummy' not in col]]
    data_observation = data_observation.drop(['y'],axis=1)
    data_observation = y_q.join(data_observation,how='outer')
    
    return data_observation

data_observation = get_dummy_us_growth_endog()

data_index = data_observation.index
data_columns = data_observation.columns
## end
########


#######
#### C.2.0 Initialization

# initialize model parameters
phi=dict(
    param_cycle_design_betas = [1] * (len(data_columns) - 1),
    param_cycle_ar_betas = [0,0],
    param_innov1_ar_betas = [0,0],
    param_innov_ar_betas =  [[0,0] for i in range(1,len(data_columns))], # ρ
    param_var_trend = 0.001,
    param_var_stoch_cycle_var = 0.01,
    param_var_stoch_innov_var = 0.01
)    

# initialize stochastic vols
stoch_vol=dict(
    stoch_vol_cycle = [0.1] * (len(data_index)-OBS_INITIAL_AR_BURN),
    
    #  stochastic volitility for idiosyncratic component of growth (nobs * 1)
    stoch_vol_innov = [0.1] * (len(data_index)-OBS_INITIAL_AR_BURN),
    
    #  stochastic volitility for idiosyncratic component of cycle obsercation (nobs * (m-1)) 
    stoch_vol_obs_innov = [[0.1] * (len(data_index)-OBS_INITIAL_AR_BURN) for i in range(len(data_columns)-1)]
)

#### end



########
## C.2.1 Draw latent variables conditional on model parameters and SVs

# initialize endog
def derive_ar_diff_y_minus_intercept(data_observation, data_intercept, innovation_ar_params):
    n_lags = len(innovation_ar_params[0])
    data_cycle_n_innovation = data_observation - data_intercept
    
    data_y_ar_diff = data_cycle_n_innovation.copy()
    for i in range(n_lags):
        data_obs_m_ar = data_cycle_n_innovation.multiply([l[i] for l in innovation_ar_params])
        data_y_ar_diff-= data_obs_m_ar.shift(i+1)

    fvidx = data_y_ar_diff.first_valid_index()
    data_y_ar_diff.loc[:fvidx] = data_y_ar_diff.loc[:fvidx].bfill()
    return data_y_ar_diff


data_intercept = pd.DataFrame(0, index=data_index, columns=data_columns)
endog = data_observation.copy()
endog.iloc[:,1:] = derive_ar_diff_y_minus_intercept(data_observation.iloc[:,1:], data_intercept.iloc[:,1:], phi['param_innov_ar_betas'])

endog = endog.iloc[OBS_INITIAL_AR_BURN:]
atitg_model = ATITG(endog, phi, stoch_vol)

atitg_simsmoother = atitg_model.simulation_smoother()
atitg_simsmoother.simulate()

                # atitg_model.update([], phi=phi, stoch_vol=stoch_vol)
                # atitg_filterred = atitg_model.filter([])
                # atitg_smoothed = atitg_model.smooth([])
                
                # atitg_filterred.filtered_state
                # atitg_smoothed.smoothed_state
                # atitg_simsmoother.simulated_state
                # plt.plot(atitg_smoothed.smoothed_state[0])

## end

########
## C.2.2 Draw the variance of the time-varying GDP growth component
states = atitg_simsmoother.simulated_state
phi['param_var_trend'] = draw_posterior_sigma2(X=states[[0], :-1].T, Y=states[0, 1:], beta=np.array([1]), prior_params=(1, 0.001))
    
## end

########
## C.2.3 Draw the autoregressive parameters of the factor VAR
states = atitg_simsmoother.simulated_state
nobs = atitg_simsmoother.simulated_state.shape[1]
error_cov = np.zeros([nobs]*2)
np.fill_diagonal(error_cov, stoch_vol['stoch_vol_cycle'])

phi['param_cycle_ar_betas'] = draw_beta_gls(X=states[[6,7],:].T, Y=states[5,:], error_cov=error_cov, reject_explosive=True) # the 5th state is the cycle factor

## end


########
## C.2.4 Draw the factor loadings
# vol of innovation is approximated from innovation AR process error
states = atitg_simsmoother.simulated_state
nobs = atitg_simsmoother.simulated_state.shape[1]
X = states[[5],:].T

for i in range(1, data_observation.shape[1]):
    Y = np.array(data_observation.iloc[OBS_INITIAL_AR_BURN:,i])
    vols = stoch_vol['stoch_vol_obs_innov'][i-1]    
    innov_ar_betas = phi['param_innov_ar_betas'][i-1]
    error_cov = np.zeros([nobs]*2)
    np.fill_diagonal(error_cov, vols)
    rng = np.arange(nobs-2)
    error_cov[rng,rng+1] = np.array(vols[:-2]) * innov_ar_betas[0]
    error_cov[rng,rng+2] = np.array(vols[:-2]) * innov_ar_betas[1]
    error_cov[rng+1,rng] = np.array(vols[:-2]) * innov_ar_betas[0]
    error_cov[rng+2,rng] = np.array(vols[:-2]) * innov_ar_betas[1]
    
    error_cov[-1,-2] = vols[-2] * innov_ar_betas[0]
    error_cov[-2,-1] = vols[-2] * innov_ar_betas[0]
        
    phi['param_cycle_design_betas'][i-1] = draw_beta_gls(X=X, Y=Y, error_cov=error_cov)[0] # the 5th state is the cycle factor

## end
########


########
## C.2.5 Draw the serial correlation coefficients of the idiosyncratic components
states = atitg_simsmoother.simulated_state
nobs = atitg_simsmoother.simulated_state.shape[1]
f = states[[5],:].T

for i in range(1, data_observation.shape[1]):
    y = np.array(data_observation.iloc[OBS_INITIAL_AR_BURN:,i])
    vols = stoch_vol['stoch_vol_obs_innov'][i-1]
    factor_loading_beta = phi['param_cycle_design_betas'][i-1]
    idio = y - f.dot(np.array([factor_loading_beta]))
    
    X = np.vstack((idio[1:-1], idio[:-2])).T
    Y = np.array(idio[2:])

    error_cov = np.zeros([nobs-2]*2)
    np.fill_diagonal(error_cov, vols[2:])
        
    phi['param_innov_ar_betas'][i-1] = draw_beta_gls(X=X, Y=Y, error_cov=error_cov) # the 5th state is the cycle factor

## end
########



###### unit test #######
def test_derive_ar_diff_y_minus_intercept():
    data_observation = pd.DataFrame([[1,2,3],[2,3,1],[3,2,1]],columns=['a','b','c'])
    data_intercept = pd.DataFrame([[1,1,1],[1,1,1],[1,1,1]],columns=['a','b','c'])
    innovation_ar_params = [[0],[0.5],[1]]
    data_y_ar_diff = derive_ar_diff_y_minus_intercept(data_observation, data_intercept, innovation_ar_params)
    print(data_observation)
    print(data_intercept)
    print(innovation_ar_params)
    print(data_y_ar_diff)
    assert data_y_ar_diff.sum().sum() == 2.5
##########
    