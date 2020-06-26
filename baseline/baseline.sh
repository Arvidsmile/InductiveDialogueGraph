#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --partition=gpu_shared
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=slurm_output/SwDA-baseline.out

module purge
module load 2019
module load Anaconda3/2018.12

. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate Stellarenvironment

srun python kerasTest.py --dataset SwDA --kfold 10 --epochs 20
