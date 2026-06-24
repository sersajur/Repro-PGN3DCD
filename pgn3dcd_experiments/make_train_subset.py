"""Create a symlinked subset of HKCD train scenes for fast debug iteration.

Symlinks (not copies) the chosen scene folders into a new directory, so it
costs no extra disk. Point training at it with a hydra override:
    data.dataTrainFile=<dest_subset_dir>/

Symlinks are RELATIVE, so they keep resolving when the dataset is mounted at a
different path inside a container (e.g. host /mnt/data/dataset/HKCD vs container
/data/HKCD) — as long as <src_train_dir> and <dest_subset_dir> stay siblings.

Usage:
    python make_train_subset.py <src_train_dir> <dest_subset_dir> <scene> [<scene> ...]

    <src_train_dir>    Existing dir holding all scene folders (e.g. /data/HKCD/Train)
    <dest_subset_dir>  Dir to create with symlinks to the chosen scenes
    <scene> ...        One or more scene folder names to include
"""
import os
import os.path as osp
import sys


def main(src_train_dir, dest_subset_dir, scenes):
    os.makedirs(dest_subset_dir, exist_ok=True)
    for scene in scenes:
        src = osp.join(src_train_dir, scene)
        if not osp.isdir(src):
            sys.exit(f"ERROR: scene not found: {src}")
        link = osp.join(dest_subset_dir, scene)
        if osp.islink(link) or osp.exists(link):
            os.remove(link)
        # Relative target so the link survives host->container path remapping.
        rel_src = osp.relpath(src, dest_subset_dir)
        os.symlink(rel_src, link)
        print(f"linked {scene} -> {rel_src}")

    print(f"\nSubset of {len(scenes)} scenes created at {dest_subset_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])