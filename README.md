# mtDNA LSR Benchmarking for MitoDetective paper
Simulated long-read (ONT) mitochondrial DNA datasets with large-scale rearrangements (LSRs) at defined heteroplasmy levels, used to benchmark two pipelines on detection accuracy, heteroplasmy estimation and resource usage.

---
Workflow overview
```
rCRS reference
   │
   ▼
1. generate_mtdna_variants.py   → mutant FASTAs + truth manifests
   │
   ▼
2. simulate_ont_reads.sh        → pooled WT/mutant molecules → PBSIM3 ONT reads → aligned BAMs
   │
   ▼
   Run mitoSAlt and MitoDetective on each BAM (timed with psrecord)
   │
   ├─▶ 3. evaluate_sv_calls.py      → detection + heteroplasmy accuracy vs truth
   ├─▶ 4. max_col5_split.py         → per-arc max deletion burden from burden tables
   └─▶ 5. compare_psrecord.py       → CPU / memory / runtime comparison
```
---
1. Variant generation
`generate_mtdna_variants.py` takes the rCRS reference (NC_012920.1, 16,569 bp) and writes one mutant FASTA per variant.
Variants modelled:
Category	Examples
Single deletions	Common 4977 bp deletion (8470–13447), large major-arc, small minor-arc
Origin-spanning deletion	15000 → 600 (tests circular handling)
Duplications	Minor-arc tandem dup; small and large origin-spanning dups
Multiple deletions	Two mixtures of 6 and 12 overlapping major/minor-arc deletions with varying weights, modelling multiple-deletion disorders
Compound	Common deletion + complementary duplication (33:67)
Deletion dimer	Retained arc of the common deletion repeated in tandem
Outputs: `wildtype.fa`, `wildtype_doubled.fa` (circularity workaround), a `mutant.fa` per simple variant, per-component FASTAs plus `components.tsv` (weights) for mixtures, `variant_manifest.tsv` and `heteroplasmy_levels.txt`.
2. Read simulation
`simulate_ont_reads.sh` (SGE job run by wrapper script) builds a pool of 200 molecules per heteroplasmy level (5, 10, 25, 50, 75%), mixing mutant and wildtype copies. For mixtures, mutant molecules are shared out across components by weight using largest-remainder rounding.
Reads are simulated with PBSIM3, aligned to chrM with minimap2 (`map-ont`) and sorted/indexed with samtools.
Output: `simulated_reads/ont/<variant>/het{005,010,025,050,075}.bam`
3. Accuracy evaluation
`run_comparison.py` scores pipeline calls against a truth table (`sample`, `svtype`, `start`, `end`, `wraps_origin`, `effective_heteroplasmy`). It supports two output styles:
`per_cluster_het` – one call table with a heteroplasmy per call (MitoSAlt's output)
`cluster_plus_burden` – per-sample cluster tables plus a positional deletion-burden table (MitoDetective's output)
Key steps:
Match calls to truth by reciprocal overlap with circular-genome awareness; samples with no calls are scored as all-FN
Report pooled, per-sample, macro-averaged, per-SV-type and read-weighted precision/recall/F1
Sweep minimum read-support thresholds
Heteroplasmy accuracy on TPs (Bland–Altman, scatter), burden-curve and sample-level total burden error (samples with overlapping truth deletions excluded where not meaningful)
`--compare-with` lines up two pipelines' per-sample tables side by side
Requires the helper modules `matcher.py`, `sv_plots.py` and `resource_and_het_metrics.py`.
4. Per-arc maximum burden
`max_col5_split.py` recursively finds files ending in a given suffix and reports the maximum of column 5 (burden/heteroplasmy) for positions (column 2) up to and after a threshold (default 5500), giving a quick minor-arc vs major-arc summary.
5. Resource usage
Each pipeline run is recorded with `psrecord`. `compare_psrecord.py` pairs logs from the two pipelines by sample name (ignoring underscores and timestamps) and plots CPU and memory over time, plus overview charts of peak/average memory, runtime and peak CPU, and a delta table.
```bash
python compare_psrecord.py \
  --program-a logs/pipelineA --program-b logs/pipelineB \
  --name-a PipelineA --name-b PipelineB \
  -o resource_comparison.png --csv resource_summary.csv
```
---
Dependencies
Python 3 with Biopython, pandas, numpy and matplotlib; PBSIM3, minimap2, samtools, psrecord.
