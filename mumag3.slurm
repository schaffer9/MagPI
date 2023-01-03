#!/bin/bash
#SBATCH --partition=gpusims
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G	# MB
#SBATCH --time=1-0		# MINUTES, DAYS-HOURS
#
# OPTIONAL
#SBATCH --gres=gpu
#SBATCH --export=None
#SBATCH --job-name="mumag3"	# default is batch script name
#SBATCH --exclude=Nd		# do not grant resouces on those nodes

EPOCHS=8000
INCREASE_PENALTY_AFTER=800
SAMPLES_DOM=13
SAMPLES_BND=11
SAMPLES_CONV=14
NUM_WEIGHTS=8
WEIGHTS_SCALE=4
MAG_LAYERS=3
MAG_UNITS=100
PENALTY_INCREASE=2
LR_START=1e-5
LR_END=1e-7
BATCH_SIZE=100
ALPHA0=1.
OUTPUT_DIR=mumag3-test1

# echo "JOBNAME:" $SLURM_JOB_NAME
# echo "PARTITION:" $SLURM_JOB_PARTITION
# echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
# echo "ALLOC. NODES:" $SLURMD_NODENAME
# echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR

# echo "Activate conda env"
# source /promethium/home/programs/miniconda3/bin/activate
# conda activate ML
# echo "Activated"

echo "Run python mumag3.py --samples_dom=$SAMPLES_DOM --samples_bnd=$SAMPLES_BND --samples_conv=$SAMPLES_CONV --number_weights=$NUM_WEIGHTS --weights_scale=$WEIGHTS_SCALE --mag_layers=$MAG_LAYERS --mag_units=$MAG_UNITS --epochs=$EPOCHS --increase_penalty_after=$INCREASE_PENALTY_AFTER --lr_start=$LR_START --lr_end=$LR_END --batch_size=$BATCH_SIZE --alpha0=$ALPHA0 --output_dir=$OUTPUT_DIR"
python mumag3.py --samples_dom=$SAMPLES_DOM --samples_bnd=$SAMPLES_BND --samples_conv=$SAMPLES_CONV --number_weights=$NUM_WEIGHTS --weights_scale=$WEIGHTS_SCALE --mag_layers=$MAG_LAYERS --mag_units=$MAG_UNITS --epochs=$EPOCHS --increase_penalty_after=$INCREASE_PENALTY_AFTER --lr_start=$LR_START --lr_end=$LR_END --batch_size=$BATCH_SIZE --alpha0=$ALPHA0 --output_dir=$OUTPUT_DIR 
echo "Completed"
