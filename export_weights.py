"""Make a small, self-contained weights file from a training checkpoint.

A training checkpoint also holds the optimizer state, so it is about three times bigger than the model alone.
The exported file holds only the model weights (float16 by default), the model config and the tokenizer, and
generate.py can load it directly.

Example:
    python export_weights.py --ckpt best.pt --out tinystories_25m_fp16.pt
    python generate.py --ckpt tinystories_25m_fp16.pt --prompt "Once upon a time"
"""
import argparse
import os

import torch


def export(ckpt_path, out_path, tokenizer_path=None, dtype="float16"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target = {"float16": torch.float16, "float32": torch.float32}[dtype]
    converted, shared = {}, {}
    for name, t in ck["model"].items():
        key = t.data_ptr()                           # tied weights share storage, keep them shared in the file
        if key not in shared:
            shared[key] = t.detach().to(target).clone()
        converted[name] = shared[key]
    payload = {"model": converted, "cfg": ck["cfg"], "dtype": dtype, "best_val": ck.get("best_val"), "iter": ck.get("iter")}
    if "chars" in ck:
        payload["chars"] = ck["chars"]
    else:
        path = tokenizer_path or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"tokenizer not found at {path}. Pass --tokenizer path/to/tokenizer.json")
        payload["tokenizer_json"] = open(path, encoding="utf-8").read()
    torch.save(payload, out_path)
    return os.path.getsize(ckpt_path), os.path.getsize(out_path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=None, help="tokenizer.json for a BPE checkpoint (default: next to the checkpoint)")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    a = p.parse_args(argv)
    before, after = export(a.ckpt, a.out, a.tokenizer, a.dtype)
    print(f"{a.ckpt}: {before / 2**20:.1f} MB -> {a.out}: {after / 2**20:.1f} MB ({a.dtype})")


if __name__ == "__main__":
    main()
