#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=slurm_output

module purge
module load 2019
module load Anaconda3/2018.12

. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate Stellarenvironment

srun python kerasTest.py \
--dataset MANtIS \
--batch_size 32 \
--epochs 20 \
--kfold 10 \
--num_test 20 \
--testing 1 \
--start_node 0 \
--aggregator maxpool \
--expname lisatest
