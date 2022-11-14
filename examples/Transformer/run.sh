#!/bin/bash
#SBATCH --job-name=mup_original
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-10%4
#SBATCH --output=/network/scratch/n/normandf/mup_original/logs/slurm-%A_%a.out
#SBATCH --error=/network/scratch/n/normandf/mup_original/logs/slurm-%A_%a.err

module load miniconda/3
conda activate $SCRATCH/conda/mup

export EXP_NAME=${EXP_NAME:-"mup_original"}

echo "Starting sweep with name $EXP_NAME"

orion hunt -n $EXP_NAME --exp-max-broken=999 --exp-max-trials=1000 \
    python main.py --load_base_shapes width256.bsh --optimizer adam --cuda \
    --d_model~"choices(128,256,512,1024,2048,4096)" \
    --lr~"loguniform(1e-5,1e-1,default_value=0.01)"
