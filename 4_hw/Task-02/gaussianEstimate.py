import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def branin_hoo(x):
    """Calculate the Branin-Hoo function value for given input."""
    a=1
    b=5.1/(4*np.pi**2) 
    c=5./np.pi
    r=6
    s=10
    t=1./(8*np.pi)
    
    return (a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 +
            s * (1 - t) * np.cos(x[0]) + s)

# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF kernel."""
    sqdist = np.sum((x1[:, None] - x2[None, :])**2, axis=2)
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matérn kernel (nu=1.5)."""
    dists = np.linalg.norm(x1[:, None] - x2[None, :], axis=2)
    sqrt3_d = np.sqrt(3) * dists / length_scale
    return sigma_f**2 * (1 + sqrt3_d) * np.exp(-sqrt3_d)

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    sqdist = np.sum((x1[:, None] - x2[None, :])**2, axis=2)
    return sigma_f**2 * (1 + sqdist / (2 * alpha * length_scale**2))**(-alpha)

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """Compute the log-marginal likelihood."""
    y_train = y_train.reshape(-1, 1)
    y_mean = np.mean(y_train)
    y_train_centered = y_train - y_mean

    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    L = np.linalg.cholesky(K)

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train_centered))
    lml = -0.5 * y_train_centered.T @ alpha
    lml -= np.sum(np.log(np.diag(L)))
    lml -= 0.5 * len(x_train) * np.log(2 * np.pi)

    return lml.item()

def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""
    best_lml = -np.inf
    best_params = (1.0, 1.0, noise)
    y_train = y_train.reshape(-1, 1)

    for length_scale in [0.1, 0.5, 1.0, 2.0]:
        for sigma_f in [0.5, 1.0, 2.0]:
            lml = log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise)
            if lml > best_lml:
                best_lml = lml
                best_params = (length_scale, sigma_f, noise)
    return best_params

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction. Assumes y_train is centered."""
    x_train = np.atleast_2d(x_train)
    x_test = np.atleast_2d(x_test)
    y_train = y_train.reshape(-1, 1)

    # Compute kernel matrices
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    K_s = kernel_func(x_train, x_test, length_scale, sigma_f)
    K_ss = kernel_func(x_test, x_test, length_scale, sigma_f) + 1e-8 * np.eye(len(x_test))

    # Compute posterior mean and covariance
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    cov = K_ss - v.T @ v
    std = np.sqrt(np.maximum(np.diag(cov), 0))  # Ensure non-negative variance

    return mu.ravel(), std


# Acquisition Functions (Simplified, no erf)
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = (mu - y_best - xi) / (sigma + 1e-8)
    Phi = 1 / (1 + np.exp(-1.702 * z))
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return (mu - y_best - xi) * Phi + sigma * phi

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    z = (mu - y_best - xi) / (sigma + 1e-8)
    Phi = 1. / (1. + np.exp(-1.702 * z))
    phi = Phi
    phi[sigma < 1e-8] = 0.0
    return Phi

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""
    plt.figure(figsize=(8, 6))

    cp = plt.contourf(x1_grid, x2_grid, z_values, levels=50, cmap='viridis')

    plt.colorbar(cp)
    plt.scatter(x_train[:, 0], x_train[:, 1], c='red', marker='x', label='Training Points')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement,
        'Random' : None
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    # print(f"True min: {true_values.min()}, max: {true_values.max()}")

    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            y_mean_val = y_train.mean()
            y_train_centered = y_train - y_mean_val
            y_train_centered = y_train_centered.reshape(-1, 1)

            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train_centered, kernel_func)

            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()

                # Center current y
                y_mean_val = y_train_current.mean()
                y_train_centered = y_train_current - y_mean_val
                y_train_centered = y_train_centered.reshape(-1, 1)

                # Predict on test grid
                y_mean, y_std = gaussian_process_predict(
                    x_train_current, y_train_centered, x_test,
                    kernel_func, length_scale, sigma_f, noise
                )
                y_mean += y_mean_val  # Add mean back

                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)

                # Acquisition logic
                if acq_name is not None:
                    if acq_name == 'Random':
                        random_idx = np.random.choice(len(x_test))
                        new_x = x_test[random_idx]
                    else:
                        y_best = np.min(y_train_current)
                        acq_values = acq_func(y_mean, y_std, y_best)
                        best_acq_idx = np.argmax(acq_values)
                        new_x = x_test[best_acq_idx]

                    new_y = branin_hoo(new_x)

                    x_train_current = np.vstack([x_train_current, new_x])
                    y_train_current = np.append(y_train_current, new_y)

                    # Re-center and retrain
                    y_mean_val = y_train_current.mean()
                    y_train_centered = y_train_current - y_mean_val
                    y_train_centered = y_train_centered.reshape(-1, 1)

                    length_scale, sigma_f, noise = optimize_hyperparameters(x_train_current, y_train_centered, kernel_func)

                    y_mean, y_std = gaussian_process_predict(
                        x_train_current, y_train_centered, x_test,
                        kernel_func, length_scale, sigma_f, noise
                    )
                    y_mean += y_mean_val  # Re-add mean after prediction

                    y_mean_grid = y_mean.reshape(x1_grid.shape)
                    y_std_grid = y_std.reshape(x1_grid.shape)

                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()