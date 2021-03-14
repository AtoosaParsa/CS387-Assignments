import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt

pyro.set_rng_seed(101)

def model1(t):
  return 2

def  model2(t):
  return 3 #pyro.sample("m2", pyro.distributions.Normal(10.0, 1.0))

def discrete_obs_switching_model(data, model1, model2, T, N):
  log_scale = pyro.sample("loc", pyro.distributions.Normal(0.0, 1.0))
  z = pyro.sample("z_0", pyro.distributions.Normal(0.0, torch.exp(log_scale)))
  retval = torch.empty((T, N))

  for t in pyro.markov(range(1, T)):
    z = pyro.sample(f"z_{t}", pyro.distributions.Normal(z, 1.0))
    #invlogit
    p = torch.exp(z)/(1+torch.exp(z))

    with pyro.plate("data_plate", size=N):
      switch = pyro.sample(f"switch_{t}", pyro.distributions.Bernoulli(p))
      #if switch ==0:
      #  y = model1(t)
      #else:
      #  y = model2(t)
      y = torch.mul(switch, model1(t)) + torch.mul((1-switch), model2(t))
      #with pyro.poutine.mask(mask = (switch == 1)):
      #  y=model1(t)
      #with pyro.poutine.mask(mask = (switch == 0)):
      #  y=model2(t)
      x = pyro.sample(f"x_{t}", pyro.distributions.Poisson(y), obs=None if data==None else data[:, t])
      retval[t-1] = x
  return  retval

# model is fully generative here, generate some data
data = discrete_obs_switching_model(None, model1, model2, 100, 10)
print(data)

# trace
static_trace = discrete_obs_switching_model(None, model1, model2, 2, 2)
print(static_trace.nodes)

# print the trace, from https://willcrichton.net/notes/probabilistic-programming-under-the-hood/
print({
    name: {
        'value': props['value'],
        'prob': props['fn'].log_prob(props['value']).exp()
    }
    for (name, props) in static_trace.nodes.items()
    if props['type'] == 'sample'
})
