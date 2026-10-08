#!/usr/bin/env bash
VARIANTS_DIR="$1"
 
for variant_dir in "$VARIANTS_DIR"/*/; do
    [[ -d "$variant_dir" ]] || continue
    variant_dir="${variant_dir%/}"  
    echo "submitting runs for $variant_dir"
    qsub ./create_simulated_reads_ONT.sh "$variant_dir"
    #qsub ./create_simulated_reads_PB.sh "$variant_dir"
    echo "runs sent for $variant_dir"
done

echo "all runs submitted"


# bash run_create_simulated_reads.sh /home/skgtth3/Scratch/skgtth3/mitosalt/mtdna_test_variants/DEL_common_4977