#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu-2080ti
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_swav
#SBATCH --time=01:00:00
#SBATCH --mem=100G

#SBATCH --output=logfiles/%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logfiles/%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sar.mueller@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 


DATASET_PATH="/mnt/qb/work/berens/smueller93/"
EXPERIMENT_PATH="/mnt/qb/work/berens/smueller93/swav/experiments/cifar10_500ep_bs512_pretrain/eval_linear/"
mkdir -p $EXPERIMENT_PATH

python_path=/mnt/qb/work/berens/smueller93/miniconda3/envs/swav/bin/python 

srun singularity exec --bind $WORK --nv cuda10_container.sif $python_path -u predict.py