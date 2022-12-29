#!/bin/bash
#SBATCH --partition=gpusims
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G	# MB
#SBATCH --time=1-0		# MINUTES, DAYS-HOURS
#
# OPTIONAL
#SBATCH --export=None
#SBATCH --job-name="mumag3"	# default is batch script name
#SBATCH --exclude=Nd		# do not grant resouces on those nodes

EPOCHS=2000
SAMPLES_DOM=13
MAG_LAYERS=3
MAG_UNITS=50
LR=1e-5
BATCH_SIZE=100

echo "JOBNAME:" $SLURM_JOB_NAME
echo "PARTITION:" $SLURM_JOB_PARTITION
echo "SUBMIT HOST:" $SLURM_SUBMIT_HOST
echo "ALLOC. NODES:" $SLURMD_NODENAME
echo "SUBMIT DIR:" $SLURM_SUBMIT_DIR

echo "Activate conda env"
source /promethium/home/programs/miniconda3/bin/activate
conda activate ML
echo "Activated"

echo "Run python mumag3_init_mag.py --samples_dom=$SAMPLES_DOM --mag_layers=$MAG_LAYERS --mag_units=$MAG_UNITS --epochs=$EPOCHS  --lr=$LR --batch_size=$BATCH_SIZE --alpha0=$ALPHA0"
python mumag3_init_mag.py --samples_dom=$SAMPLES_DOM --mag_layers=$MAG_LAYERS --mag_units=$MAG_UNITS --epochs=$EPOCHS  --lr=$LR --batch_size=$BATCH_SIZE --alpha0=$ALPHA0
echo "Completed"
