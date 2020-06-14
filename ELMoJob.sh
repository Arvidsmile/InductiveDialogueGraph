#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --nodes=1
#SBATCH --partition=gpu_short
#SBATCH --mail-user=arvid.lindstrom@student.uva.nl
#SBATCH --output=test-ELMoGPUSwDA.out

module purge
module load 2019
module load Anaconda3/2018.12

conda activate ELMoenvironment

cp -r $HOME/CSVData "$TMPDIR"

srun python generateELMO.py --dataset SwDA --testing 1 --step_size 100
cp "$TMPDIR"/ELMo_SwDA.pickle $HOME/ELMoPickled
