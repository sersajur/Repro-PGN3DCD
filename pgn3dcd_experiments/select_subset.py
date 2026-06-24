"""Pick a class-balanced subset of HKCD train scenes.

Exhaustively searches subsets of a given size for the one whose aggregate
change ratio is closest to the full-split ratio, so the debug subset preserves
the original change/unchanged balance.

Usage:
    python select_subset.py <subset_size>

Per-scene (change, total) counts are embedded from scene_class_stats.py output
on the 15-scene HKCD Train split (global ratio = 0.0907).
"""
import sys
from itertools import combinations

# (scene_name, n_change, n_total) — measured from raw cd_type labels.
SCENES = [
    ("11-NE-11A",   570570,   5436013),
    ("11-NE-11B",  1989854,   5847454),
    ("11-NE-11C",   860074,   1919665),
    ("11-NE-11D",  1506427,   2555075),
    ("11-NE-12A",   421447,   7842811),
    ("11-NE-12D",   237464,   8632147),
    ("11-NE-1B",    197935,   6281080),
    ("11-NE-1D",    405928,  11601595),
    ("11-NE-6A",    352457,   8207556),
    ("11-NE-6B",    587261,   8902563),
    ("11-NE-6C",    397250,   7421237),
    ("11-NE-6D",    655925,   7113390),
    ("11-NE-7A",    493127,   8388118),
    ("11-NE-7C",    378845,   5831861),
    ("11-NE-7D",    460416,   8982109),
]

GLOBAL_CHANGE = sum(s[1] for s in SCENES)
GLOBAL_TOTAL = sum(s[2] for s in SCENES)
GLOBAL_RATIO = GLOBAL_CHANGE / GLOBAL_TOTAL


def main(k):
    best = []
    for combo in combinations(SCENES, k):
        c = sum(s[1] for s in combo)
        t = sum(s[2] for s in combo)
        r = c / t
        best.append((abs(r - GLOBAL_RATIO), r, t, [s[0] for s in combo]))
    best.sort()

    print(f"Global ratio = {GLOBAL_RATIO:.4f}  over {GLOBAL_TOTAL:,} points\n")
    print(f"Top 5 subsets of size {k} closest to global change ratio:\n")
    for dr, r, t, names in best[:5]:
        pct = 100 * t / GLOBAL_TOTAL
        print(f"  ratio={r:.4f} (Δ={dr:+.4f})  points={t:>11,} ({pct:4.1f}% of full)")
        print(f"     {', '.join(names)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(int(sys.argv[1]))