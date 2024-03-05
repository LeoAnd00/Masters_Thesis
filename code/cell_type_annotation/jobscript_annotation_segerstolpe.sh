#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 14:00:00
#SBATCH --gpus-per-node=A100:4

apptainer exec /cephyr/users/leoan/Alvis/Masters_Thesis/MScEnv.sif python benchmark_annotation.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/data_for_evaluating_cell_type_annotation/Segerstolpe.h5ad' 'trained_models/Assess_generalisability/Segerstolpe/' 'benchmark_results/Segerstolpe/Benchmark_results' '' 'Segerstolpe'