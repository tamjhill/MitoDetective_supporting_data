#!/usr/bin/env python3
"""
generate_mtdna_variants.py

Generates mtDNA variant FASTAs at multiple heteroplasmy levels for SV benchmarking.
Outputs:
  - A doubled reference FASTA (for circularity handling)
  - Per-variant, per-heteroplasmy FASTA files (mixed wildtype + mutant molecules)

Usage:
    python generate_mtdna_variants.py --ref rCRS.fa --outdir mtdna_variants

"""

import argparse
import os
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

MT_LEN = 16569  # ref length


VARIANTS = [
    # --- Large deletions ---
    {
        "name": "DEL_common_4977",
        "type": "DEL",
        "start": 8470,
        "end": 13447,
        "notes": "Common deletion (mtDNA4977), most frequent pathogenic mtDNA deletion"
    },
    {
        "name": "DEL_major_large",
        "type": "DEL",
        "start": 6000,
        "end": 15000,
        "notes": "Large arc deletion spanning most of the major arc"
    },
    {
        "name": "DEL_minor_small",
        "type": "DEL",
        "start": 1500,
        "end": 2500,
        "notes": "Small deletion minor arc deletion"
    },
    {
        "name": "DEL_origin_spanning",
        "type": "DEL_CIRC",
        "start": 15000,
        "end": 16569 + 600,
        "notes": "Origin-spanning deletion, tests circular genome handling"
    },

    # --- Duplications ---
    {
        "name": "DUP_origin_spanning_small",
        "type": "DUP",
        "start": 15500,
        "end": 16569 + 1200,
        "notes": "Small duplication, wrapping origin"
    },
    {
        "name": "DUP_origin_spanning_large",
        "type": "DUP",
        "start": 15000,
        "end": 16569 + 5000,
        "notes": "Large duplication, wrapping origin"
    },
        {
        "name": "DUP_minor_only",
        "type": "DUP",
        "start": 2000,
        "end": 4500,
        "notes": "Duplication in the monor arc only"
    },

    # --- Multiple deletions ---
    {
        "name": "MULTI_DEL_version1",
        "type": "MIXTURE",
        "components": [
            {"label": "del1_a", "svtype": "DEL", "start": 5700, "end": 11800, "weight": 0.08},
            {"label": "del1_b", "svtype": "DEL", "start": 6800, "end": 12000, "weight": 0.40},
            {"label": "del1_c", "svtype": "DEL", "start": 7300, "end": 10800, "weight": 0.10},
            {"label": "del1_d", "svtype": "DEL", "start": 7800, "end": 11000, "weight": 0.20},
            {"label": "del1_e", "svtype": "DEL", "start": 8300, "end": 13200, "weight": 0.15},
            {"label": "del1_f", "svtype": "DEL", "start": 7500, "end": 13900, "weight": 0.07},

        ],
        "notes": "Deletions scattered across major arc, varying in size, varying weight, models multiple deletion disorders"
    },
            {
        "name": "MULTI_DEL_version2",
        "type": "MIXTURE",
        "components": [
            {"label": "del2_a", "svtype": "DEL", "start": 1500, "end": 3800, "weight": 0.05},
            {"label": "del2_b", "svtype": "DEL", "start": 2000, "end": 4000, "weight": 0.08},
            {"label": "del2_c", "svtype": "DEL", "start": 6000, "end": 10800, "weight": 0.07},
            {"label": "del2_e", "svtype": "DEL", "start": 6400, "end": 11000, "weight": 0.06},
            {"label": "del2_f", "svtype": "DEL", "start": 7100, "end": 13200, "weight": 0.08},
            {"label": "del2_g", "svtype": "DEL", "start": 5900, "end": 12800, "weight": 0.18},
            {"label": "del2_h", "svtype": "DEL", "start": 7800, "end": 11500, "weight": 0.07},
            {"label": "del2_i", "svtype": "DEL", "start": 8300, "end": 13400, "weight": 0.10},
            {"label": "del2_j", "svtype": "DEL", "start": 8100, "end": 11800, "weight": 0.06},
            {"label": "del2_k", "svtype": "DEL", "start": 8900, "end": 14600, "weight": 0.06},
            {"label": "del2_l", "svtype": "DEL", "start": 9200, "end": 15000, "weight": 0.12},
            {"label": "del2_m", "svtype": "DEL", "start": 9700, "end": 14200, "weight": 0.07},
        ],
        "notes": "Deletions scattered across major arc, varying in size, varying weight, models multiple deletion disorders"
    },
    
    # --- Complex arrangements ---
    {
        "name": "DEL_and_DUP_33_67",
        "type": "MIXTURE",
        "components": [
            {
                "label": "del",
                "svtype": "DEL",
                "start": 8470,
                "end": 13448,
                "weight": 0.33,
            },
            {
                "label": "dup",
                "svtype": "DUP",
                "start": 13448,
                "end": 16569 + 8469,
                "weight": 0.67,
            },
        ],
        "notes": "Compound SV: common deletion + duplication"
    },
        {
        "name": "DEL_and_DUP_50_50",
        "type": "MIXTURE",
        "components": [
            {
                "label": "del",
                "svtype": "DEL",
                "start": 8470,
                "end": 13448,
                "weight": 0.50,
            },
            {
                "label": "dup",
                "svtype": "DUP",
                "start": 13448,
                "end": 16569 + 8469,
                "weight": 0.50,
            },
        ],
        "notes": "Compound SV: common deletion + duplication, split het"
    },
    {
        "name": "DEL_DIMER_common",
        "type": "DEL_DIMER",
        "del_start": 8470,
        "del_end": 13448,
        "notes": (
            "Deletion dimer: retained arc (13448-16569 + 0-8469) duplicated in tandem, "
            "no other bases. Models dimer of deleted molecules."
        )
    },
]

HETEROPLASMY_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def apply_deletion(seq, start, end):
    return seq[:start] + seq[end:]


def apply_circular_deletion(seq, start, end):
    """end > MT_LEN signals origin-spanning; deletes seq[start:] and seq[0:end-MT_LEN]."""
    wrap_end = end - MT_LEN
    return seq[wrap_end:start]


def apply_tandem_duplication(seq, start, end):
    segment = seq[start:end]
    return seq[:end] + segment + seq[end:]


def apply_circular_duplication(seq, start, end, mt_len):
    """end > mt_len signals origin-spanning duplication."""
    wrap_end = end - mt_len
    segment = seq[start:] + seq[:wrap_end]
    return seq[:wrap_end] + segment + seq[wrap_end:]


def build_mutant_sequence(variant, ref_seq):
    vtype = variant["type"]
    if vtype == "DEL":
        return apply_deletion(ref_seq, variant["start"], variant["end"])
    elif vtype == "DEL_CIRC":
        return apply_circular_deletion(ref_seq, variant["start"], variant["end"])
    elif vtype == "DUP":
        if variant["end"] > MT_LEN:
            return apply_circular_duplication(ref_seq, variant["start"], variant["end"], MT_LEN)
        return apply_tandem_duplication(ref_seq, variant["start"], variant["end"])
    elif vtype == "DEL_DIMER":
        retained_arc = ref_seq[variant["del_end"]:] + ref_seq[:variant["del_start"]]
        return retained_arc + retained_arc
    elif vtype == "MIXTURE":
        raise RuntimeError("Call build_component_sequence for MIXTURE components")
    else:
        raise ValueError(f"Unknown variant type: {vtype}")


def build_component_sequence(component, ref_seq):
    svtype = component["svtype"]
    if svtype == "DEL":
        return apply_deletion(ref_seq, component["start"], component["end"])
    elif svtype == "DUP":
        if component["end"] > MT_LEN:
            return apply_circular_duplication(ref_seq, component["start"], component["end"], MT_LEN)
        return apply_tandem_duplication(ref_seq, component["start"], component["end"])
    else:
        raise ValueError(f"Unsupported component svtype: {svtype}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
 
def write_fasta(seq, seq_id, description, path):
    record = SeqRecord(Seq(seq), id=seq_id, description=description)
    SeqIO.write(record, path, "fasta")


def write_variant(variant, ref_seq, ref_id, outdir):
    name = variant["name"]
    vtype = variant["type"]
    var_dir = os.path.join(outdir, name)
    os.makedirs(var_dir, exist_ok=True)

    print(f"\n  {name}  ({vtype})")
    print(f"    {variant['notes']}")

    if vtype == "MIXTURE":
        total_weight = sum(c["weight"] for c in variant["components"])
        if not (0.999 <= total_weight <= 1.001):
            raise ValueError(
                f"{name}: component weights sum to {total_weight:.4f}, must sum to 1.0"
            )
        print(f"    Component weights sum: {total_weight:.3f} (OK)")
        for c in variant["components"]:
            mut_seq = build_component_sequence(c, ref_seq)
            out_path = os.path.join(var_dir, f"{c['label']}_mutant.fa")
            write_fasta(mut_seq, f"{ref_id}_{name}_{c['label']}",
                        f"{c['svtype']} weight={c['weight']:.3f}", out_path)
            print(f"    [{c['label']}] weight={c['weight']:.3f}  len={len(mut_seq)}bp  -> {out_path}")
        # Write manifest — simulation scripts use weight to split the mutant fraction
        manifest_path = os.path.join(var_dir, "components.tsv")
        with open(manifest_path, "w") as f:
            f.write("label\tsvtype\tweight\tmutant_fa\n")
            for c in variant["components"]:
                f.write(f"{c['label']}\t{c['svtype']}\t{c['weight']}\t{c['label']}_mutant.fa\n")
        print(f"    Manifest: {manifest_path}")
 
    else:
        mut_seq = build_mutant_sequence(variant, ref_seq)
        size_change = len(mut_seq) - len(ref_seq)
        out_path = os.path.join(var_dir, "mutant.fa")
        write_fasta(mut_seq, f"{ref_id}_{name}", vtype, out_path)
        if vtype == "DEL_DIMER":
            arc_len = len(mut_seq) // 2
            print(f"    arc={arc_len}bp  dimer={len(mut_seq)}bp")
        else:
            print(f"    size change: {size_change:+d}bp  mutant length: {len(mut_seq)}bp")
        print(f"    -> {out_path}")
 
 
def write_manifest(variants, outdir, het_levels):
    path = os.path.join(outdir, "variant_manifest.tsv")
    with open(path, "w") as f:
        f.write("variant_name\ttype\tnotes\n")
        for v in variants:
            f.write(f"{v['name']}\t{v['type']}\t{v['notes']}\n")
    print(f"\n  Manifest: {path}")
 
    het_path = os.path.join(outdir, "heteroplasmy_levels.txt")
    with open(het_path, "w") as f:
        for h in het_levels:
            f.write(f"{h}\n")
    print(f"  Het levels: {het_path}")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Generate mtDNA variant FASTAs for SV benchmarking"
    )
    parser.add_argument("--ref", required=True,
                        help="Path to rCRS reference FASTA (NC_012920.1)")
    parser.add_argument("--outdir", default="mtdna_variants",
                        help="Output directory (default: mtdna_variants)")
    args = parser.parse_args()
 
    os.makedirs(args.outdir, exist_ok=True)
 
    records = list(SeqIO.parse(args.ref, "fasta"))
    if len(records) != 1:
        raise ValueError(f"Expected 1 sequence in {args.ref}, got {len(records)}")
    ref_seq = str(records[0].seq).upper()
    ref_id = records[0].id
    if len(ref_seq) != MT_LEN:
        print(f"  Warning: reference length is {len(ref_seq)}, expected {MT_LEN}")
    print(f"Reference: {ref_id}  ({len(ref_seq)}bp)")
 
    # Write wildtype reference
    wt_path = os.path.join(args.outdir, "wildtype.fa")
    write_fasta(ref_seq, ref_id, "wildtype reference", wt_path)
    print(f"Wildtype:  {wt_path}")
 
    # Write doubled wildtype for circularity workaround
    wt_doubled_path = os.path.join(args.outdir, "wildtype_doubled.fa")
    write_fasta(ref_seq + ref_seq, ref_id + "_doubled",
                "doubled wildtype for circularity workaround", wt_doubled_path)
    print(f"Wildtype (doubled): {wt_doubled_path}")
 
    print("\nGenerating variant FASTAs...")
    for variant in VARIANTS:
        write_variant(variant, ref_seq, ref_id, args.outdir)
 
    write_manifest(VARIANTS, args.outdir, HETEROPLASMY_LEVELS)
 
    print("\nDone.")
 
if __name__ == "__main__":
    main()
