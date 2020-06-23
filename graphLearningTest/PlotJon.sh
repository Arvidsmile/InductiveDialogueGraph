#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --partition=gpu_shared
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=noPCA-plot-swda.out

module purge
module load 2019
module load Anaconda3/2018.12

. /sw/arch/Debian9/EB_production/2019/software/Anaconda3/2018.12/etc/profile.d/conda.sh
conda activate ELMoenvironment


srun python newELMo.py
