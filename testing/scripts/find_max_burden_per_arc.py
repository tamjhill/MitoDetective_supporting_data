#!/usr/bin/env python3

import sys
from pathlib import Path


def get_max_col5_split(filepath, threshold):
    max_before = None
    max_after = None

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            row = line.rstrip('\n').split('\t')
            if len(row) < 5:
                continue
            try:
                col2 = float(row[1])
                col5 = float(row[4])
            except ValueError:
                continue  # skip header or non-numeric rows

            if col2 <= threshold:
                if max_before is None or col5 > max_before:
                    max_before = col5
            else:
                if max_after is None or col5 > max_after:
                    max_after = col5

    return max_before, max_after


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python max_col5_split.py <root_folder> <suffix> [threshold]")
        sys.exit(1)

    root_folder = Path(sys.argv[1])
    suffix = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 5500

    if not root_folder.is_dir():
        print(f"Error: {root_folder} is not a valid directory")
        sys.exit(1)

    matches = sorted(root_folder.rglob(f"*{suffix}"))

    if not matches:
        print("No matching files found.")
        return

    print(f"filename\tmax_col5_upto_{int(threshold)}\tmax_col5_after_{int(threshold)}")
    for filepath in matches:
        max_before, max_after = get_max_col5_split(filepath, threshold)
        before_str = max_before if max_before is not None else "N/A"
        after_str = max_after if max_after is not None else "N/A"
        print(f"{filepath}\t{before_str}\t{after_str}")


if __name__ == "__main__":
    main()
