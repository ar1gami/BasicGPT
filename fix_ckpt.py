"""Rebuild a PyTorch checkpoint file from a folder that holds its unzipped contents.

A .pt file made by torch.save is a zip archive. If a download tool unzipped it, you get a folder
(containing data.pkl, a data folder and so on) where the file should be. This packs it back.

    py fix_ckpt.py best.pt best_fixed.pt
"""
import os
import sys
import zipfile


def repack(folder, out_path):
    if not os.path.isdir(folder):
        raise SystemExit(f"{folder} is not a folder")
    if not os.path.exists(os.path.join(folder, "data.pkl")):
        raise SystemExit(f"{folder} has no data.pkl, so it is not an unzipped checkpoint")
    prefix = "archive"                                   # torch only needs one top level name
    names = []
    for root, _, files in os.walk(folder):
        for f in files:
            full = os.path.join(root, f)
            names.append((full, prefix + "/" + os.path.relpath(full, folder).replace(os.sep, "/")))
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as z:
        for full, arc in sorted(names, key=lambda t: (t[1] != prefix + "/data.pkl", t[1])):
            z.write(full, arc)
    return len(names)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    n = repack(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]} from {n} files")
