#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl

module purge
module load 2019
module load Anaconda3/2018.12

. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate Stellarenvironment

srun python main.py \
--dataset MANtIS \
--batch_size 256 \
--epochs 10 \
--kfold 4 \
--num_test 10 \
--testing 1 \
--start_node 0 \
--aggregator maxpool \
--model_size c3 \
--expname finaltest
