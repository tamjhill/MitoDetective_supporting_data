#!/bin/bash 

module purge
module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate pbsim3_env

echo "starting run"
python3 /home/skgtth3/Scratch/skgtth3/mitosalt/generate_chrM_variants.py --ref /home/skgtth3/Scratch/skgtth3/resources/hg38_chrM_human.fa --outdir mtdna_variants
echo "run complete"