#!/usr/bin/env python
import itertools
import subprocess
import multiprocessing
import csv

datasets = ['moons', 'circles', 'manycircles', 'blobs', 'helix']
sizes = [8000, 8000, 8000, 8000, 10000]
dimensions = [2, 2, 2, 2, 3]
num_classes = [2, 2, 8, 2, 2]
schedulers = ['linear', 'sigmoid', 'cosine'] # To update
lbetas = [0.005]
ubetas = [0.05]
n_steps = [10, 50, 100, 150, 200] # To update
lrs = [0.05]
batch_sizes = [100]
guidance_scales = [0.2, 0,3, 0.5, 0.8, 1.0]
reward_scales = [0.5] # To keep for part 3
num_gpus = 5  # There are 5 GPUs

def run_experiment(params):
    """Runs a single experiment."""
    dataset, size, n_dim, n_classes, scheduler, lbeta, ubeta, n_step, lr, batch_size, guidance_scale, reward_scale, gpu_id = params
    results_file = f"results_{dataset}_cond.csv"

    # Open CSV in append mode
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if file is empty
        if f.tell() == 0:
            writer.writerow(["Dataset", "Scheduler", "Lbeta", "Ubeta", "Steps", "LR", "Batch Size", "Guidance Scale", "NLL Score", "Accuracy"])

        print(f"Running on GPU {gpu_id}: {dataset}, {scheduler}, beta: {lbeta} {ubeta}, T: {n_step}, LR: {lr}, Batch: {batch_size}")

        cmd_train = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python ddpm.py --mode train "
            f"--dataset {dataset} --n_classes {n_classes} --epochs 30 --n_dim {n_dim} --n_samples {size} "
            f"--scheduler {scheduler} --batch_size {batch_size} --n_steps {n_step} --reward_scale {reward_scale} "
            f"--lbeta {lbeta} --ubeta {ubeta} --lr {lr} --guidance_scale {guidance_scale} "
        )
        subprocess.run(cmd_train, shell=True)
        
        # Extract NLL score (Modify parsing based on output format)
        cmd_sample = cmd_train.replace("train", "sample")
        process = subprocess.run(cmd_sample, shell=True, capture_output=True, text=True)
        nll_score = process.stdout.strip().split()[-1] if process.stdout else "N/A"

        # Extract Classification
        cmd_classify = cmd_train.replace("train", "classify")
        process = subprocess.run(cmd_classify, shell=True, capture_output=True, text=True)
        accuracy = process.stdout.strip().split()[-1] if process.stdout else "N/A"

        # Write result to CSV
        writer.writerow([dataset, scheduler, lbeta, ubeta, n_step, lr, batch_size, guidance_scale, reward_scale, nll_score, accuracy])
        f.flush()  # Ensure immediate write

def main():
    all_params = []
    gpu_cycle = itertools.cycle(range(num_gpus))  # Cycle through GPUs dynamically

    # Create all experiment configurations
    for dataset, size, n_dim, n_classes in zip(datasets, sizes, dimensions, num_classes):
        for scheduler, (lbeta, ubeta), n_step, lr, batch_size, (guidance_scale, reward_scale) in \
        itertools.product(
            schedulers, zip(lbetas, ubetas), n_steps, lrs, batch_sizes, zip(guidance_scales, reward_scales)
        ):
            gpu_id = next(gpu_cycle)  # Assign GPU dynamically
            all_params.append((dataset, size, n_dim, n_classes, scheduler, lbeta, ubeta, n_step, lr, batch_size, guidance_scale, reward_scale, gpu_id))

    # Use Pool with `processes=None` to use all available CPU cores
    with multiprocessing.Pool(processes=min(len(all_params), num_gpus * 4)) as pool:
        pool.map(run_experiment, all_params)

if __name__ == "__main__":
    main()
