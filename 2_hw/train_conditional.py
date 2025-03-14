#!/usr/bin/env python
import itertools
import subprocess
import multiprocessing
import csv

datasets = ['moons', 'circles', 'manycircles', 'blobs', 'helix']
sizes = [8000, 8000, 8000, 8000, 10000]
dimensions = [2, 2, 2, 2, 3]
num_classes = [2, 2, 8, 2, 2]
schedulers = ['linear', 'sigmoid', 'cosine']
lbetas = [0.001, 0.005, 0.01]
ubetas = [0.02, 0.1, 0.2]
n_steps = [10, 50, 100, 150, 200]
lrs = [0.1, 0.01]
batch_sizes = [100, 200]
num_gpus = 5  # There are 5 GPUs

def run_experiment(params):
    """Runs a single experiment."""
    dataset, size, n_dim, n_classes, scheduler, lbeta, ubeta, n_step, lr, batch_size, gpu_id = params
    results_file = f"results_{dataset}_cond.csv"

    # Open CSV in append mode
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if file is empty
        if f.tell() == 0:
            writer.writerow(["Dataset", "Scheduler", "Lbeta", "Ubeta", "Steps", "LR", "Batch Size", "NLL Score"])

        print(f"Running on GPU {gpu_id}: {dataset}, {scheduler}, beta: {lbeta} {ubeta}, T: {n_step}, LR: {lr}, Batch: {batch_size}")

        cmd_train = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python ddpm.py --mode train "
            f"--dataset {dataset} --n_classes {n_classes} --epochs 30 --n_dim {n_dim} --n_samples {size} "
            f"--scheduler {scheduler} --batch_size {batch_size} --n_steps {n_step} "
            f"--lbeta {lbeta} --ubeta {ubeta} --lr {lr}"
        )
        
        cmd_sample = cmd_train.replace("train", "sample")

        subprocess.run(cmd_train, shell=True)
        process = subprocess.run(cmd_sample, shell=True, capture_output=True, text=True)

        # Extract NLL score (Modify parsing based on output format)
        nll_score = process.stdout.strip().split()[-1] if process.stdout else "N/A"

        # Write result to CSV
        writer.writerow([dataset, scheduler, lbeta, ubeta, n_step, lr, batch_size, nll_score])
        f.flush()  # Ensure immediate write

def main():
    all_params = []
    gpu_cycle = itertools.cycle(range(num_gpus))  # Cycle through GPUs dynamically

    # Create all experiment configurations
    for dataset, size, n_dim, n_classes in zip(datasets, sizes, dimensions, num_classes):
        for scheduler, (lbeta, ubeta), n_step, lr, batch_size in itertools.product(
            schedulers, zip(lbetas, ubetas), n_steps, lrs, batch_sizes
        ):
            gpu_id = next(gpu_cycle)  # Assign GPU dynamically
            all_params.append(\
                    (dataset, size, n_dim, n_classes, scheduler, lbeta, ubeta, n_step, lr, batch_size, gpu_id))

    # Use Pool with `processes=None` to use all available CPU cores
    with multiprocessing.Pool(processes=min(len(all_params), num_gpus * 4)) as pool:
        pool.map(run_experiment, all_params)

if __name__ == "__main__":
    main()
