import torch
import torch.utils.data
from torch import nn
import os
import dataset
import utils
from ddpm import DDPM, NoiseScheduler, train

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = 'exps/albatross_ddpm'
    os.makedirs(run_name, exist_ok=True)

    n_steps = 150
    beta_start, beta_end = 0.005, 0.05
    epochs, batch_size, lr = 100, 64, 1e-3
    n_dim = 3  # Adjust based on the dataset
    
    # Load dataset
    data_X, data_y = dataset.load_dataset("albatross")
    data_X, data_y = data_X.to(device), data_y.to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_X, data_y), batch_size=batch_size, shuffle=True)
    
    # Initialize model and scheduler
    model = DDPM(n_dim=n_dim, n_steps=n_steps).to(device)
    noise_scheduler = NoiseScheduler(num_timesteps=n_steps, beta_start=beta_start, beta_end=beta_end)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)
    
if __name__ == "__main__":
    main()
