#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=test-ELMoGPUSwDA.out

module purge
module load 2019
module load Anaconda3/2018.12

. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate ELMoenvironment


srun python generateELMO.py --dataset SwDA --testing 1 --step_size 100
