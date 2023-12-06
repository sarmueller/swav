#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu-2080ti
#SBATCH --gres=gpu:2 
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --job-name=swav_cifar10_500ep_bs512_pretrain
#SBATCH --time=72:00:00
#SBATCH --mem=150G

#SBATCH --output=logfiles/%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logfiles/%j.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sar.mueller@uni-tuebingen.de   # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID 

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000
echo $master_node
echo $dist_url


DATASET_PATH="/mnt/qb/work/berens/smueller93/"
EXPERIMENT_PATH="/mnt/qb/work/berens/smueller93/swav/experiments/cifar10_500ep_bs512_pretrain"
mkdir -p $EXPERIMENT_PATH

python_env_path = /mnt/qb/work/berens/smueller93/miniconda3/envs/swav/bin/python 

srun singularity exec --bind $WORK --nv cuda10_container.sif $python_env_path -u main_swav.py \
--data_path $DATASET_PATH \
--epochs 500 \
--batch_size 256 \
--size_crops 32 \
--nmb_crops 2 \
--nmb_prototypes 30 \
--temperature 0.5 \
--queue_length 0 \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 900 \
--dist_url $dist_url \
--arch resnet50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH