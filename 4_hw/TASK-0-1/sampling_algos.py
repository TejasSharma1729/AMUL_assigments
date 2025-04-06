import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from get_results import EnergyRegressor
from time import clock_gettime
from sklearn.manifold import TSNE
# You can import any other torch modules you need below #

##########################################################

# Other settings
DEVICE = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

FEAT_DIM = 784 # Input dimension
OUTPUT_PATH = './'  # Path to save the output files

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:
    """
    Algorithm 1 (from PDF) for MCMC sampling.
    __init__ method initializes the model and parameters.
    __call__ method performs the sampling process.
    """
    def __init__(self, model, burn_in=1000, lr=0.01):
        """
        @param model: The energy model E_theta.
        @param burn_in: Number of burn-in steps, default is 100.
        @param num_steps: Number of sampling steps, default is 100.
        @param lr: Learning rate for the sampling process, default is 0.01.
        No return value.
        """
        self.model = model.to(DEVICE)
        self.burn_in = burn_in
        self.lr = lr
    
    def __call__(self, num_samples=1000):
        """
        #@param num_samples: Number of samples M to generate, default is 100.
        #@return: numpy array of shape (M, FEAT_DIM) containing M generated samples,
                    burn-in time and completion time.
        """
        burn_in_time = 0.0
        completion_time = 0.0
        t0 = clock_gettime(0)
        # Initialize start time and burn-in time and completion time

        x = torch.randn((1, FEAT_DIM), device=DEVICE)
        samples = []
        # Initialize x0 with random noise
        
        for time_step in range(self.burn_in + num_samples):
            x.requires_grad = True
            e_x = self.model(x)
            e_x.backward()
            g = x.grad
            # Compute the gradient of E_theta(x) w.r.t x
            x.grad.data.zero_()

            xi = torch.randn_like(x).to(DEVICE)
            # Sample noise xi from N(0, I)

            sqrt_lr = torch.sqrt(Tensor([self.lr])).to(DEVICE)
            x_dash = x - 0.5 * self.lr * g  +  sqrt_lr * xi
            # Propose a new sample x_dash

            e_x_dash = self.model(x_dash)
            e_x_dash.backward()
            g_dash = x.grad
            # Compute the gradient of E_theta(x_dash) w.r.t x (not x_dash)
            x.grad.data.zero_()

            log_q_x_x_dash = - 0.25 * self.lr * torch.norm(x - x_dash + 0.5 * self.lr * g_dash) ** 2
            log_q_x_dash_x = - 0.25 * self.lr * torch.norm(x_dash - x + 0.5 * self.lr * g) ** 2
            exp_term = e_x - e_x_dash + log_q_x_x_dash - log_q_x_dash_x
            one = torch.Tensor([1.0]).to(DEVICE)
            alpha = torch.min(one, torch.exp(exp_term))
            # Compute the acceptance probability alpha

            u = torch.rand(1, device=DEVICE)
            # Uniformly sample u from [0, 1] for acceptance-rejection
            if (u < alpha).item():
                x = x_dash
                # Accept the proposed sample
            else:
                x = x
                # Reject the proposed sample and keep the current sample
            
            x = x.detach().requires_grad_()
            # Detach x and set requires_grad to True for the next iteration

            if time_step == self.burn_in - 1:
                burn_in_time = clock_gettime(0) - t0
                # Record the burn-in time

            if time_step >= self.burn_in:
                samples.append(x.detach().cpu())
                # Store the sample after burn-in
            
        completion_time = clock_gettime(0) - t0
        # Record the completion time

        samples_npy = torch.cat(samples, dim=0).numpy()
        return samples_npy, burn_in_time, completion_time
    
class Algo2_Sampler:
    """
    Algorithm 2 (from PDF) for MCMC sampling.
    __init__ method initializes the model and parameters.
    __call__ method performs the sampling process.
    """
    def __init__(self, model, burn_in=1000, lr=0.01):
        """
        @param model: The energy model E_theta.
        @param burn_in: Number of burn-in steps, default is 100.
        @param num_steps: Number of sampling steps, default is 100.
        @param lr: Learning rate for the sampling process, default is 0.01.
        No return value.
        """
        self.model = model.to(DEVICE)
        self.burn_in = burn_in
        self.lr = lr
    
    def __call__(self, num_samples=1000):
        """
        #@param num_samples: Number of samples M to generate, default is 100.
        #@return: numpy array of shape (M, FEAT_DIM) containing M generated samples,
                    burn-in time and completion time.
        """
        burn_in_time = 0.0
        completion_time = 0.0
        t0 = clock_gettime(0)
        # Initialize start time and burn-in time and completion time

        x = torch.randn((1, FEAT_DIM), device=DEVICE)
        samples = []
        # Initialize x0 with random noise
        
        for time_step in range(self.burn_in + num_samples):
            x.requires_grad = True
            e_x = self.model(x)
            e_x.backward()
            g = x.grad
            # Compute the gradient of E_theta(x) w.r.t x
            x.grad.data.zero_()

            xi = torch.randn_like(x).to(DEVICE)
            # Sample noise xi from N(0, I)

            sqrt_lr = torch.sqrt(Tensor([self.lr])).to(DEVICE)
            x = x - 0.5 * self.lr * g  +  sqrt_lr * xi
            # Propose a new sample x and blindly accept it
            
            x = x.detach().requires_grad_()
            # Detach x and set requires_grad to True for the next iteration

            if time_step == self.burn_in - 1:
                burn_in_time = clock_gettime(0) - t0
                # Record the burn-in time
            
            if time_step >= self.burn_in:
                samples.append(x.detach().cpu())
                # Store the sample after burn-in
        
        completion_time = clock_gettime(0) - t0
        # Record the completion time

        samples_npy = torch.cat(samples, dim=0).numpy()
        return samples_npy, burn_in_time, completion_time

    
# --- Main Execution ---
if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = './trained_model_weights.pth'  # Path to the model weights file

    model = EnergyRegressor(FEAT_DIM)
    # Initialize the model

    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    # Load the model weights

    algo1_sampler = Algo1_Sampler(model, burn_in=1000, lr=0.01)
    algo2_sampler = Algo2_Sampler(model, burn_in=1000, lr=0.01)
    # Initialize the samplers

    samples_algo1, burn_in_time_algo1, completion_time_algo1 = algo1_sampler(num_samples=1000)
    samples_algo2, burn_in_time_algo2, completion_time_algo2 = algo2_sampler(num_samples=1000)
    # Perform sampling using both algorithms

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=SEED)
    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate=200, random_state=SEED)
    # Initialize t-SNE for dimensionality reduction

    samples_algo1_2d = tsne.fit_transform(samples_algo1)
    samples_algo2_2d = tsne.fit_transform(samples_algo2)
    # Perform t-SNE on the samples (2D)

    samples_algo1_3d = tsne_3d.fit_transform(samples_algo1)
    samples_algo2_3d = tsne_3d.fit_transform(samples_algo2)
    # Perform t-SNE on the samples (3D)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # Create subplots for 2D visualization

    ax1.scatter(samples_algo1_2d[:, 0], samples_algo1_2d[:, 1], c='blue', label='Algo-1')
    ax1.set_title('Algo-1 Samples (2D)')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend()
    # Plot Algo-1 samples

    ax2.scatter(samples_algo2_2d[:, 0], samples_algo2_2d[:, 1], c='red', label='Algo-2')
    ax2.set_title('Algo-2 Samples (2D)')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.legend()
    # Plot Algo-2 samples
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'algo_samples_2d.png'))
    plt.show()
    # Save the 2D plot

    plt.close(fig)
    # Close the figure

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Create subplots for 3D visualization

    ax1.scatter(samples_algo1_3d[:, 0], samples_algo1_3d[:, 1], samples_algo1_3d[:, 2], c='blue', label='Algo-1')
    ax1.set_title('Algo-1 Samples (3D)')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.set_zlabel('t-SNE Component 3')
    ax1.legend()
    # Plot Algo-1 samples

    ax2.scatter(samples_algo2_3d[:, 0], samples_algo2_3d[:, 1], samples_algo2_3d[:, 2], c='red', label='Algo-2')
    ax2.set_title('Algo-2 Samples (3D)')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.set_zlabel('t-SNE Component 3')
    ax2.legend()
    # Plot Algo-2 samples

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'algo_samples_3d.png'))
    plt.show()
    # Save the 3D plot  

    plt.close(fig)
    # Close the figure

    # Display the burn-in and completion times
    print("--- Sampling Times ---")
    print(f"Algo-1 Burn-in Time: {burn_in_time_algo1:.4f} seconds")
    print(f"Algo-1 Completion Time: {completion_time_algo1:.4f} seconds")
    print(f"Algo-2 Burn-in Time: {burn_in_time_algo2:.4f} seconds")
    print(f"Algo-2 Completion Time: {completion_time_algo2:.4f} seconds")