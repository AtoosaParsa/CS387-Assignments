import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt

pyro.set_rng_seed(101)

def continuous_hmm(data, N, T):
  loc = pyro.sample("loc", pyro.distributions.Normal(0.0, 1.0))
  log_scale = pyro.sample("log_scale", pyro.distributions.Normal(0.0, 1.0))
  obs_scale = pyro.sample("obs_scale", pyro.distributions.Gamma(2.0, 2.0))
  retval = torch.empty((T, N))

  with pyro.plate("data_plate", size=N):
    x = pyro.sample("x_0", pyro.distributions.Normal(loc, torch.exp(log_scale)))

    for t in pyro.markov(range(1, T)):
      x = pyro.sample(f"x_{t}", pyro.distributions.Normal(loc+x, torch.exp(log_scale)))
      y = pyro.sample(f"y_{t}", pyro.distributions.Normal(x, obs_scale), obs=None if data==None else data[:, t])
      retval[t-1] = y
  return  retval

# model is fully generative here, generate some data
data = continuous_hmm(None, 100, 10)

# trace
static_trace = pyro.poutine.trace(continuous_hmm).get_trace(None, 2, 2)
static_trace.compute_log_prob()
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
