#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 20:00:00
#SBATCH --gpus-per-node=V100:1

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python run_benchmark.py '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/kidney_cells/Muto_merged.h5ad' 'trained_models/Kidney/' 'benchmarks/results/Kidney/benchmark' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/all_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' '_Kidney/'