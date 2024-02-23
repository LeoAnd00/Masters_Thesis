#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 30:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python benchmark_generalizability.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/merged/merged_all.h5ad' 'trained_models/Assess_generalisability/All_merged/' 'benchmarks/results/Generalizability/All_merged/Benchmark_results' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/processed/pathway_information/c5_pathways.json' '/cephyr/users/leoan/Alvis/Masters_Thesis/data/raw/gene2vec_embeddings/gene2vec_dim_200_iter_9_w2v.txt' '_Generalizability/All_merged/'