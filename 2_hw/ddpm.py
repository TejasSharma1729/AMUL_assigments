#!/usr/bin/env python
import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt

from utils import gaussian_kernel, get_likelihood, get_nll, get_emd, split_data, sample

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")
        # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        # Lifted from the notebook, link below
        # https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=qWw50ui9IZ5q
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def init_sigmoid_schedule(self, beta_start, beta_end, steepness=10):
        t = torch.linspace(-steepness, steepness, self.num_timesteps)
        self.betas = torch.sigmoid(t)  
        self.betas = beta_start + (beta_end - beta_start) * self.betas  
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def init_cosine_schedule(self, beta_start, beta_end):
        t = torch.linspace(0, torch.pi / 2, self.num_timesteps)
        self.betas = torch.cos(t) ** 2  
        self.betas = beta_start + (beta_end - beta_start) * (1 - self.betas) 
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def __len__(self):
        return self.num_timesteps
    
class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. 
        `time_embed` can be learned or a fixed function as well

        """
        super(DDPM, self).__init__()
        # MLP with Leaky ReLU activation
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.t_dim = 2 # time dimension -- a hyperparameter
        self.i_dim = self.n_dim # intermediate dimention -- for LeakyReLU layer
        # intermediate dimention need not be same as n_dim, but we chose it to be same
        self.time_embed = nn.Sequential( 
                nn.Linear(1, self.t_dim), 
        )
        self.model = nn.Sequential( 
                nn.Linear(self.n_dim + self.t_dim, self.i_dim), 
                nn.LeakyReLU(0.01),
                nn.Linear(self.i_dim, self.n_dim), 
        )
        # Reason for choice of LeakyReLU: it is a good choice for regression problems
        # and it performed well in albatross dataset, but ReLU did not perform well.

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x_and_t = torch.cat([x, t_embed], dim=-1)
        noise_out = self.model(x_and_t)
        return noise_out

class ConditionalDDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `class_embed`, time_embed` and `model`,
        `class_embed` and `time_embed` can be learned or a fixed function as well
        We assume class labels are integers, from 0 onward [-1 --> NULL case]

        """
        super(ConditionalDDPM, self).__init__()
        self.n_dim = n_dim
        self.num_classes = 0 # To be updated during training, important.
        self.n_steps = n_steps
        self.c_dim = 4 # class label dimension -- a hyperparameter
        self.t_dim = 2 # time dimension -- a hyperparameter
        self.i_dim = self.n_dim # intermediate dimention -- for LeakyReLU layer
        self.class_embed = nn.Sequential(
                nn.Linear(1, self.c_dim),
        )
        self.time_embed = nn.Sequential(
                nn.Linear(1, self.t_dim),
        )
        self.model = nn.Sequential(
                nn.Linear(self.n_dim + self.c_dim + self.t_dim, self.i_dim),
                nn.LeakyReLU(0.01),
                nn.Linear(self.i_dim, self.n_dim),
        )
        # Mostly same as normal DDPM, just added class label embedding

    def forward(self, x, y, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            y: torch.Tensor, the class label tensor [batch_size]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        y_embed = self.class_embed(y.unsqueeze(-1).float())
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x_y_t = torch.cat([x, y_embed, t_embed], dim=-1)
        noise_out = self.model(x_y_t)
        return noise_out


class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler

    def __call__(self, x):
        # Just a wrapper around the predict function
        return self.predict(x)

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class labels [batch_size]
        """
        return self.predict_proba(x).argmax(axis=1)
        # Just the class "y" with the highest probability, among all classes [computed]

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilities [batch_size, n_classes]
        """
        device = x.device
        n_classes = self.model.num_classes
        # Idea: for reasonable t, epsilon_theta (eta) predicts noise different for y_opt than -1
        # and for different class y, prediction will be suboptimal (closer to for null case, y = -1)
        # since during training (y, x_t, t) will not be seen together
        sum_diffs = torch.zeros(x.size(0), n_classes, device=device)
        # Idea: sample multiple (5) times and take the sum, so as to avoid random errors

        for _ in range(5):
            # The random timestamps will mostly be different for different samples
            rand_times = torch.randint(0, self.noise_scheduler.num_timesteps // 2, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            sqrt_abar = self.noise_scheduler.sqrt_alphas_cumprod.to(device)[rand_times.unsqueeze(1)]
            sqrt_1_abar = self.noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)[rand_times.unsqueeze(1)]
            x_t = sqrt_abar * x + sqrt_1_abar * noise

            for c in range(n_classes):
                y = torch.Tensor([c] * x.size(0)).to(device)
                model_out = self.model(x_t, y, rand_times + 1)
                model_null_out = self.model(x_noisy, -1 * torch.ones_like(y), rand_times + 1)

                # Now update sum of differences for this timestamp and this classification.
                sum_diffs[c] += torch.norm(model_out - model_null_out, dim=1)
        
        return F.softmax(sum_diffs, dim=1)
        # Return: we need a function that converts these sums to probabilities, that sum to one
        # and are proportional somewhat to sums. One option: naive sum (less sharp prediction).
        # We picked softmax, since it is a good choice for classification problems.

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    model.train()
    loss_fxn = nn.MSELoss()
    for epoch in range (epochs):
        total_loss = 0
        for x, _ in dataloader:
            optimizer.zero_grad()
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.size(0),), device=x.device)
            noise = torch.randn_like(x)
            x_noisy = noise_scheduler.sqrt_alphas_cumprod.to(x.device)[t.unsqueeze(1)] * x + \
                    noise_scheduler.sqrt_one_minus_alphas_cumprod.to(x.device)[t.unsqueeze(1)] * noise

            model_out = model(x_noisy, t + 1)
            loss = loss_fxn(model_out, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))

def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the conditional model and save the model and necessary plots

    Args:
        model: ConditionalDDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    model.train()
    all_classes = set()
    loss_fxn = nn.MSELoss()
    for epoch in range (epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.size(0),), device=x.device)
            noise = torch.randn_like(x)
            x_noisy = noise_scheduler.sqrt_alphas_cumprod.to(x.device)[t.unsqueeze(1)] * x + \
                    noise_scheduler.sqrt_one_minus_alphas_cumprod.to(x.device)[t.unsqueeze(1)] * noise

            all_classes.update(y.tolist())
            rand_nulls = torch.rand(y.shape, device=device) < 0.2 # 20% of the time, we set the value to -1
            y[rand_nulls] = -1 # Set the value to -1, for NULL (unconditional training)

            model_out = model(x_noisy, y, t + 1)
            loss = loss_fxn(model_out, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    model.num_classes = len(all_classes)
    torch.save(model.state_dict(), os.path.join(run_name, "conditional_model.pth"))


@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """   
    model.eval()
    device = next(model.parameters()).device  # Get model's device
    x = [torch.randn(n_samples, model.n_dim, device=device) for _ in range(0, model.n_steps + 1)]
    
    for t in range(model.n_steps, 0, -1):
        z = torch.randn(n_samples, model.n_dim, device=device)  # Ensure z is also on the same device
        mu = (noise_scheduler.betas[t-1]) / (noise_scheduler.sqrt_one_minus_alphas_cumprod[t-1])

        eps_theta = model.forward(x[t], torch.Tensor([t] * n_samples).to(device))  # Move to correct device
        x[t-1] = noise_scheduler.sqrt_recip_alphas[t-1] * (x[t] - mu * eps_theta) \
            + torch.sqrt(noise_scheduler.posterior_variance[t-1]) * z
        
    if return_intermediate:
        return x
    return x[0]
        
@torch.no_grad()
def sampleConditional(model, n_samples, noise_scheduler, class_label, return_intermediate=False):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
        Optionally implement return_intermediate=True, will aid in visualizing the intermediate steps
    """
    model.eval()
    x = [torch.randn(n_samples, model.n_dim) for _ in range (0, model.n_steps + 1)]
    
    for t in range(model.n_steps, 0, -1):
        z = torch.randn(n_samples, model.n_dim).to(x[t].device)    
        mu = (noise_scheduler.betas[t-1]) / (noise_scheduler.sqrt_one_minus_alphas_cumprod[t-1])
        y = torch.Tensor([class_label] * n_samples).to(x[t].device)

        eps_theta = model.forward(x[t], class_labels, torch.Tensor([t] * n_samples).to(x[t].device))
        x[t-1] = noise_scheduler.sqrt_recip_alphas[t-1] * (x[t] - mu * eps_theta) + \
                torch.sqrt(noise_scheduler.posterior_variance[t-1]) * z

    if (return_intermediate):
        return x
    return x[0]

@torch.no_grad()
def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    model.eval()
    x = [torch.randn(n_samples, model.n_dim) for _ in range (0, model.n_steps + 1)]

    for t in range(model.n_steps, 0, -1):
        z = torch.randn(n_samples, model.n_dim).to(x[t].device)
        mu = (noise_scheduler.betas[t-1]) / (noise_scheduler.sqrt_one_minus_alphas_cumprod[t-1])
        y = torch.Tensor([class_label] * n_samples).to(x[t].device)
        y0 = torch.Tensor([-1] * n_samples).to(x[t].device)

        eps_theta = model.forward(x[t], y, torch.Tensor([t] * n_samples).to(x[t].device))
        eps_theta0 = model.forward(x[t], y0, torch.Tensor([t] * n_samples).to(x[t].device))

        cond_x = noise_scheduler.sqrt_recip_alphas[t-1] * (x[t] - mu * eps_theta) + \
                torch.sqrt(noise_scheduler.posterior_variance[t-1]) * z
        cond_x0 = noise_scheduler.sqrt_recip_alphas[t-1] * (x[t] - mu * eps_theta0) + \
                torch.sqrt(noise_scheduler.posterior_variance[t-1]) * z
        x[t-1] = (1 + guidance_scale) * cond_x - guidance_scale * cond_x0

    return x[0]


def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and 
                returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--lbeta", type=float, default=0.001)
    parser.add_argument("--ubeta", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dataset", type=str, default = 'circles')
    parser.add_argument("--seed", type=int, default = 42)
    parser.add_argument("--n_dim", type=int, default = 2)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' 
    # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y),
                batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.to(device)
        model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device, weights_only=False))

        samples = sample(model, args.n_samples, noise_scheduler).to(device)

        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')

        data_X, _ = dataset.load_dataset(args.dataset)
        real_samples = data_X[:args.n_samples].to(device)

        samples = samples.to(device)
        real_samples = real_samples.to(device)
        
        real_samples = real_samples.to(device)
        samples = samples.to(device)

        # emd_score = utils.get_emd(real_samples.cpu().numpy(), samples.cpu().numpy())

        # nll_score = utils.get_nll(real_samples, samples)
        nll_score = utils.get_nll(real_samples.cpu(), samples.cpu())

        print(nll_score)

    else:
        raise ValueError(f"Invalid mode {args.mode}")
