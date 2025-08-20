#!/usr/bin/env python
import torch
import numpy as np
import os
from ddpm import DDPM, NoiseScheduler
import dataset
from utils import gaussian_kernel, get_likelihood, get_nll, get_emd, split_data, sample
import torch.utils
import torch.utils.data
import utils
import dataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 100  
beta_start = 0.001
beta_end = 0.02
n_dim = 64
epochs = 100
lr = 0.01
type = 'linear'

model = DDPM(n_dim=n_dim, n_steps=T).to(device)
model.load_state_dict(torch.load("albatross.pth", \
        map_location=device, weights_only=False))

data_X, _= dataset.load_dataset("albatross")
real_samples = data_X[:32561].to(device)
real_samples = real_samples.to(device)

with torch.no_grad():
    model.eval()
    prior_samples = torch.tensor(np.load("data/albatross_prior_samples.npy"), dtype=torch.float32, device=device)
    n_samples = prior_samples.shape[0]

    noise_scheduler = NoiseScheduler(num_timesteps=T, beta_start=beta_start, beta_end=beta_end, type=type)

    device = next(model.parameters()).device 
    x = [torch.zeros_like(prior_samples) for _ in range(T + 1)]
    x[T] = prior_samples  

    for t in range(model.n_steps, 0, -1):
        z = torch.zeros_like(prior_samples)
        mu = (noise_scheduler.betas[t-1]) / (noise_scheduler.sqrt_one_minus_alphas_cumprod[t-1])

        eps_theta = model.forward(x[t], torch.full((n_samples,), t, device=device, dtype=torch.float32))
        x[t-1] = noise_scheduler.sqrt_recip_alphas[t-1] * (x[t] - mu * eps_theta) + \
                torch.sqrt(noise_scheduler.posterior_variance[t-1]) * z
        
    samples = x[0]
    samples = samples.to(device)

    nll_score = utils.get_nll(real_samples.cpu(), samples.cpu(), 1)

    print(nll_score)
    
    subsample_size = 600
    emd_list = []
    for i in range(5):
        subsample_X = utils.sample(data_X, size=subsample_size).cpu().numpy()
        subsample_samples = utils.sample(samples, size=subsample_size).cpu().numpy()
        emd = utils.get_emd(subsample_X, subsample_samples)
        print(f'{i} EMD w.r.t train split: {emd: .3f}')
        emd_list.append(emd)

    print(f" ---------------------------------")
    print(f"Average EMD w.r.t train split: {np.mean(emd_list):.3f} ± {np.std(emd_list):.3f}")

    torch.save(samples.cpu().numpy(), "albatross_samples_reproduce.npy")
    print("Reproduced samples saved to albatross_samples_reproduce.npy")
