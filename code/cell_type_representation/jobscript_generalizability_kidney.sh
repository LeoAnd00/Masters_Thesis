#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python benchmark_generalizability.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/kidney_cells/Muto_merged.h5ad' 'trained_models/Assess_generalisability/Kidney/' 'benchmarks/results/Generalizability/Kidney/Benchmark_results' '_Generalizability/Kidney/'