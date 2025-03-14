#!/bin/bash
lbetas=(0.001 0.005 0.01)
ubetas=(0.02 0.1 0.2)
n_steps=(10 50 100 150 200)
lr=(0.1 0.01)
batch_size=(100 200)
scheduler=('linear' 'sigmoid' 'cosine')
dataset=('moons' 'circles' 'manycircles' 'blobs' 'helix')
size=(8000 8000 8000 8000 10000)
dimensions=(2 2 2 2 3)

for i in {0..4}; do for j in {0..3}; do for k in {0..1}; do for l in {0..1}; do for m in {0..2}; do for n in {0..4}; do
	echo -n "dataset: ${dataset[n]} scheduler: ${scheduler[m]} beta: ${lbetas[j]} ${ubetas[j]} T: ${n_steps[i]} lr: ${lr[k]} batch_size: ${batch_size[l]} NLL score: " >> results.txt
    CUDA_VISIBLE_DEVICE=3 python ddpm.py --dataset ${dataset[n]} --mode train --epochs 30 --n_dim 2 --n_samples ${size[n]} --scheduler ${scheduler[m]} --batch_size ${batch_size[l]} --n_steps ${n_steps[i]} --lbeta ${lbetas[j]} --ubeta ${ubetas[j]} --lr ${lr[k]}
    CUDA_VISIBLE_DEVICE=3 python ddpm.py --dataset ${dataset[n]} --mode sample --epochs 30 --n_dim 2 --n_samples ${size[n]} --scheduler ${scheduler[m]} --batch_size ${batch_size[l]} --n_steps ${n_steps[i]} --lbeta ${lbetas[j]} --ubeta ${ubetas[j]} --lr ${lr[k]} >> results.txt
done done done done done done
