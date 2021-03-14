import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt

pyro.set_rng_seed(101)

def normal_density_estimation(data, size=1):
  loc = pyro.sample("loc", pyro.distributions.Normal(0.0, 1.0))
  inverse_scale = pyro.sample("inverse_scale", pyro.distributions.Gamma(3.0, 2.0))
  with pyro.plate("data_plate", size=size):
    obs = pyro.sample("data", pyro.distributions.Normal(loc, 1.0/inverse_scale), obs=data)
  return obs

# model is fully generative here, generate some data
data = normal_density_estimation(None, size=100)

# plot the histogram
n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Data')
plt.ylabel('Empirical Probability Density')

# trace
static_trace = pyro.poutine.trace(normal_density_estimation).get_trace(None, size=1)
static_trace.compute_log_prob()
print(static_trace.nodes)
print({
    name: {
        'value': props['value'],
        'prob': props['fn'].log_prob(props['value']).exp()
    }
    for (name, props) in static_trace.nodes.items()
    if props['type'] == 'sample'
})
