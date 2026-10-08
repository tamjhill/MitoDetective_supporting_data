#!/bin/bash -l
#$ -l h_rt=2:0:0
#$ -l mem=10G
#$ -l tmpfs=20G
#$ -N synth_reads_ONT
#$ -pe smp 8
#$ -wd /home/skgtth3/Scratch/skgtth3/mitosalt

module purge
module load python/miniconda3/24.3.0-0
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate pbsim3_env

VARIANT_DIR=$1

ONT_MODEL="/home/skgtth3/.conda/envs/pbsim3_env/data/QSHMM-ONT-HQ.model"
REF="mtdna_variants/wildtype.fa"
REFMT="/home/skgtth3/Scratch/skgtth3/resources/hg38_chrM_human.fa"
OUTDIR="simulated_reads/ont"
THREADS=8
N_MOLECULES_SIMPLE=200     # total molecules in pool; het fraction = n_mutant / N_MOLECULES
N_MOLECULES_MIXTURE=200
DEPTH_PER_MOLECULE=5
HET_LEVELS=(0.05 0.10 0.25 0.50 0.75)

variant_name=$(basename "${VARIANT_DIR%/}")
out_dir="${OUTDIR}/${variant_name}"
tmp_dir="${out_dir}/tmp"
mkdir -p "$out_dir" "$tmp_dir"


# ---------------------------------------------------------------------------
# build a combined FASTA with n_mut copies of mutant and n_wt of wt
make_pool() {
    local mut_fa="$1" n_mut="$2" n_wt="$3" out_fa="$4"
    > "$out_fa"
    for (( i=0; i<n_mut; i++ )); do cat "$mut_fa" >> "$out_fa"; done
    for (( i=0; i<n_wt;  i++ )); do cat "$REF"    >> "$out_fa"; done
}


# Detect MIXTURE or single-mutant variant
# ---------------------------------------------------------------------------
is_mixture=false
[[ -f "${VARIANT_DIR}/components.tsv" ]] && is_mixture=true

if $is_mixture; then
    N_MOLECULES=$N_MOLECULES_MIXTURE
else
    N_MOLECULES=$N_MOLECULES_SIMPLE
fi
DEPTH=$DEPTH_PER_MOLECULE
echo "Using N_MOLECULES=${N_MOLECULES}  DEPTH=${DEPTH} per molecule"

# pool is 3x genome length per molecule, so divide by 3 to keep true per-molecule depth constant
#DEPTH=$(echo "$DEPTH_PER_MOLECULE / 3" | bc)
#echo "Using N_MOLECULES=${N_MOLECULES}  DEPTH=${DEPTH} per molecule"



# Build for each heteroplasmy level
# ---------------------------------------------------------------------------

for het in "${HET_LEVELS[@]}"; do
    het_pct=$(printf "%03d" "$(echo "$het * 100" | bc | cut -d. -f1)")
    out_bam="${out_dir}/het${het_pct}.bam"
    pool_fa="${tmp_dir}/het${het_pct}_pool.fa"
    echo "Building het=${het} pool..."

    n_mut=$(echo "$N_MOLECULES * $het / 1" | bc)
    n_wt=$(( N_MOLECULES - n_mut ))

    if $is_mixture; then
        # Allocate n_mut molecules across components using largest-remainder rounding
        > "$pool_fa"
        mapfile -t components < <(tail -n +2 "${VARIANT_DIR}/components.tsv")

        # Build label/weight arrays and compute allocation
        labels=()
        weights=()
        for row in "${components[@]}"; do
            labels+=("$(echo "$row"  | cut -f1)")
            weights+=("$(echo "$row" | cut -f3)")
        done
 
        # Join weights as a comma-separated string for safe passing into Python
        weights_csv=$(IFS=,; echo "${weights[*]}")
 
        allocation=$(python3 -c "
            n_mut = $n_mut
            weights = [float(w) for w in '$weights_csv'.split(',')]
            raw = [n_mut * w for w in weights]
            floors = [int(x) for x in raw]
            remainders = [x - f for x, f in zip(raw, floors)]
            deficit = n_mut - sum(floors)
            # Give the extra molecules to the components with the largest remainders
            order = sorted(range(len(weights)), key=lambda i: -remainders[i])
            for i in order[:deficit]:
                floors[i] += 1
            print(' '.join(str(f) for f in floors))
            ")
        read -ra n_comp_arr <<< "$allocation"
 
        for idx in "${!labels[@]}"; do
            label="${labels[$idx]}"
            n_comp="${n_comp_arr[$idx]}"
            fa="${VARIANT_DIR}/${label}_mutant.fa"
            for (( i=0; i<n_comp; i++ )); do cat "$fa" >> "$pool_fa"; done
        done
        for (( i=0; i<n_wt; i++ )); do cat "$REF" >> "$pool_fa"; done
    else
        make_pool "${VARIANT_DIR}/mutant.fa" "$n_mut" "$n_wt" "$pool_fa"
    fi

    # Simulate reads from pool
    echo "Simulating ONT reads (het=${het})..."
    pbsim \
        --strategy wgs \
        --method qshmm \
        --qshmm "$ONT_MODEL" \
        --depth "$DEPTH" \
        --length-mean 10000 \
        --length-sd 8000 \
        --length-max 33000 \
        --accuracy-mean 0.98 \
        --genome "$pool_fa" \
        --prefix "${tmp_dir}/het${het_pct}"

    # Align all output FQs to reference -> sorted BAM
    echo "Aligning (het=${het})..."
    zcat ${tmp_dir}/het${het_pct}_*.fq.gz \
        | minimap2 -ax map-ont -Y -t "$THREADS" "$REFMT" - \
        | samtools sort -@ "$THREADS" -o "$out_bam"
    samtools index "$out_bam"

    # Clean up per-het intermediates
    rm -f "$pool_fa" \
          ${tmp_dir}/het${het_pct}_*.fq.gz \
          ${tmp_dir}/het${het_pct}_*.maf.gz \
          ${tmp_dir}/het${het_pct}_*.ref

    echo "  Written: $out_bam"
done

rm -rf "$tmp_dir"
echo "Done: $variant_name (ONT)"
