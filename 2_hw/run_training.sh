#!/bin/bash
lbetas=(0.001 0.001 0.005 0.005 0.01)
ubetas=(0.01 0.02 0.05 0.1 0.2)
n_steps=(10 50 100 150 200)
lr=(0.1 0.05 0.01)
batch_size=(100 200)

for i in {0..4}; do for j in {0..4}; do for k in {0..2}; do for l in {0..1}; do
	echo -n "beta: ${lbetas[j]} to ${ubetas[j]} over T = ${n_steps[i]} steps, with learning_rate ${lr[k]} and batch_size ${batch_size[l]} -- NLL score: " >> results.txt
    CUDA_VISIBLE_DEVICE=3 python ddpm.py --dataset moons --mode train --epochs 20 --n_dim 2 --n_samples 8000 --batch_size ${batch_size[l]} --n_steps ${n_steps[i]} --lbeta ${lbetas[j]} --ubeta ${ubetas[j]} --lr ${lr[k]}
    CUDA_VISIBLE_DEVICE=3 python ddpm.py --dataset moons --mode sample --epochs 20 --n_dim 2 --n_samples 8000 --batch_size ${batch_size[l]} --n_steps ${n_steps[i]} --lbeta ${lbetas[j]} --ubeta ${ubetas[j]} --lr ${lr[k]} >> results.txt
done done done done
