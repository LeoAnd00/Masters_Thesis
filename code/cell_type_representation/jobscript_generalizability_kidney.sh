#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 20:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python benchmark_generalizability.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/kidney_cells/Muto_merged.h5ad' 'trained_models/Assess_generalisability/Kidney/' 'benchmarks/results/Generalizability/Kidney/Benchmark_results' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/c5_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' '_Generalizability/Kidney/'