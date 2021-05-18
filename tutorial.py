import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

import math
import torch
import gpytorch
from matplotlib import pyplot as plt


from env.env.envs import factory


"""
Description:
5/14/2021

Date:
gpytorch examples

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""


def make_env():
    env_loader = factory.create('swellex')
    env = env_loader()
    zr, zs = np.linspace(100,200, 20), [50]
    zr = np.hstack(([0], zr))
    freq = 100
    env.add_source_params(freq, zs, zr)
    dz, zmax, dr, rmax = 0.1, 216.5, 10, 10*1e3
    env.add_field_params(dz, zmax, dr, rmax)
    folder = 'at_files/'
    fname = 'swell'
    p, pos = env.run_model('kraken', folder, fname, zr_flag=True, zr_range_flag=True)
    p = np.squeeze(p)
    full_p, full_pos = env.run_model('kraken', folder, fname, zr_flag=False, zr_range_flag=True)
    full_p = full_p[:,-1]
    return env, p, pos, full_p, full_pos
    
    


def get_training_set():
    env, p, pos, full_p, full_col_pos = make_env()
    full_z = full_col_pos.r.depth
    full_p = full_p.real
    full_p /= np.max(abs(full_p))
    print(full_p.shape, full_z.shape)
    train_x = pos.r.depth
    train_p = p.real
    train_p /= np.max(abs(train_p))
    print(train_x.shape, train_p.shape)
    return train_x, train_p, full_p, full_z
    
train_x, train_p, full_col_p, full_z = get_training_set()
train_y = deepcopy(train_p)
train_y += np.random.randn(train_y.size)*0.1
train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)


tmp_x = torch.linspace(0,1,100)
tmp_y = torch.sin(tmp_x)
print(type(train_x[0]), type(train_y[0]), type(tmp_x[0]), type(tmp_y[0]))


"""
a child of ExactGp that adds a mean_modul and covar_modul attributes 
"""

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    """
    Needs to be defined to run train
    """
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)


training_iter = 1000


# Find optimal model hyperparameters
model.train()
likelihood.train()
        

# Use the adam optimizer
# lr is "learning rate"
optimizer = torch.optim.Adam(model.parameters(), lr=1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    """
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    """
    optimizer.step()



# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.from_numpy(np.linspace(1, 200, 151))
    #test_x = torch.linspace(90, 200, 51)
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(full_z, full_col_p, 'r*')
    #ax.plot(train_x.numpy(), train_p, 'r')
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')

    #ax.scatter(train_x, train_y, color='r')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['Noise-free field', 'Observed Data', 'Mean',  'Confidence'])


plt.show()
