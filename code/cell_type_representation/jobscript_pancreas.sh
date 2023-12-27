#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 06:00:00
#SBATCH --gpus-per-node=V100:4

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python run_benchmark.py '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pancreas_cells/pancreas_1_adata.h5ad' 'trained_models/Pancreas/' 'benchmarks/results/Pancreas/benchmark' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/all_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' '_Pancreas/'