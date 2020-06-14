#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=testingELMoGPU.out

module purge
module load 2019
module load Anaconda3/2018.12

conda activate ELMoenvironment
srun python 
