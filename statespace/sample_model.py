import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm

# True model parameters
nobs = int(1e3)
true_phi = np.r_[0.5, -0.2]
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# sample MLE model with time-varying deisgn and transion matrix 
class AR2(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(AR2, self).__init__(endog, k_states=2, k_posdef=1,initialization='approximate_diffuse')

        # Setup the fixed components of the state space representation
        temp = [1]*50+[0.7]*50
        self['design']=np.array([[temp,[0]*100]])
        self['transition'] = np.array([[[0]*100, [0]*100],
                                  [[1]*50+[0.8]*50, [0]*100]])
        self['selection', 0, 0] = 1

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(AR2, self).update(params, transformed, **kwargs)

        #self['transition', 0, :] = params[:2]
        self['transition', 1, 0][:50] = params[1]        
        self['state_cov', 0, 0] = params[2]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0,0,1]  # these are very simple

# Create and fit the model
mod = AR2(endog)
res = mod.fit()
print(res.summary())