#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --partition=gpu_shared
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=slurm_out/SwDA-ELMo.out

module purge
module load 2019
module load Anaconda3/2018.12

. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate ELMoenvironment


srun python generateELMO.py --dataset SwDA --testing 0 --step_size 1
