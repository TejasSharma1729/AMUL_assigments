#!/bin/bash
lbetas=(0.001 0.001 0.005 0.005 0.01)
ubetas=(0.01 0.02 0.05 0.1 0.2)
n_steps=(10 50 100 150 200)

for i in {0..4}; do for j in {0..4}; do
    python ddpm.py --dataset moons --mode train --epochs 100 --n_dim 2 --n_samples 8000 --batch_size 10 --n_steps ${n_steps[i]} --lbeta ${lbetas[j]} --ubeta ${ubetas[j]} --lr 0.05
    python ddpm.py --dataset moons --mode sample --epochs 100 --n_dim 2 --n_samples 8000 --batch_size 10 --n_steps ${n_steps[i]} --lbeta ${lbetas[j]} --ubeta ${ubetas[j]} --lr 0.05
done done
