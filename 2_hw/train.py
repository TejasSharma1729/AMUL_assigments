#!/usr/bin/env python
import itertools
import subprocess
import multiprocessing
import csv

datasets = ['moons', 'blobs', 'circles', 'manycircles', 'helix']
sizes = [8000, 8000, 8000, 8000, 10000]
dimensions = [2, 2, 2, 2, 3]
schedulers = ['linear', 'sigmoid', 'cosine']
lbetas = [0.001]
ubetas = [0.02]
n_steps = [10, 50, 100, 150, 200]
lrs = [0.01]
batch_sizes = [100]
num_gpus = 5

def run_experiment(dataset, size, n_dim, gpu_id):
    results_file = f"results_{dataset}.csv"
    
    # Open CSV file in append mode
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header only if file is empty
        if f.tell() == 0:
            writer.writerow(["Dataset", "Scheduler", "Lbeta", "Ubeta", "Steps", "LR", "Batch Size", "NLL Score"])
        
        for scheduler, (lbeta, ubeta), n_step, lr, batch_size in itertools.product(
            schedulers, zip(lbetas, ubetas), n_steps, lrs, batch_sizes
        ):
            print(f"Running: {dataset}, {scheduler}, beta: {lbeta} {ubeta}, T: {n_step}, LR: {lr}, Batch: {batch_size}")

            cmd_train = (
                f"CUDA_VISIBLE_DEVICES=1 python ddpm.py --conditional 0 "
                f"--dataset {dataset} --mode train --epochs 30 --n_dim {n_dim} --n_samples {size} "
                f"--scheduler {scheduler} --batch_size {batch_size} --n_steps {n_step} "
                f"--lbeta {lbeta} --ubeta {ubeta} --lr {lr}"
            )
            
            cmd_sample = cmd_train.replace("train", "sample")

            subprocess.run(cmd_train, shell=True)
            process = subprocess.run(cmd_sample, shell=True, capture_output=True, text=True)

            # Extract NLL score (Modify parsing logic based on output format)
            nll_score = process.stdout.strip().split()[-1] if process.stdout else "N/A"

            # Write result to CSV
            writer.writerow([dataset, scheduler, lbeta, ubeta, n_step, lr, batch_size, nll_score])
            f.flush()  # Ensure data is written to disk

def main():
    processes = []
    for gpu_id, (dataset, size, n_dim) in enumerate(zip(datasets, sizes, dimensions), start=0):  # Start from 0
        p = multiprocessing.Process(target=run_experiment, args=(dataset, size, n_dim, gpu_id))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
