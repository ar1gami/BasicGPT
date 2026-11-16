"""Generate text from a trained checkpoint. Works with both kinds of checkpoint in this project:

  character level (made by train.py)         python generate.py --ckpt runs/shakespeare/ckpt_best.pt --prompt "ROMEO:"
  BPE (made by the Kaggle TinyStories notebook, or by export_weights.py)
                                             python generate.py --ckpt best.pt --prompt "Once upon a time"

A BPE checkpoint needs its tokenizer. It is read from --tokenizer, or from tokenizer.json in the same folder as
the checkpoint, or from the checkpoint itself when it was made by export_weights.py.
"""
import argparse
import os
import sys
import time

import torch

from llama_gpt import CharData, Config, Model

EOS_TOKEN = "<|endoftext|>"


class CharTokenizer:
    eos_id = None

    def __init__(self, chars):
        self.vocab = CharData("", chars=chars)
        self.stoi = self.vocab.stoi

    def encode(self, s):
        return self.vocab.encode(s)

    def decode(self, ids):
        return self.vocab.decode(ids)


class BPETokenizer:
    """Wraps a Hugging Face tokenizers object (byte level BPE with an end of text token)."""

    def __init__(self, tok):
        self.tok = tok
        self.eos_id = tok.token_to_id(EOS_TOKEN)

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode(ids)          # special tokens are skipped in the text


def cut_at_eos(ids, prompt_len, eos_id):
    """Drop everything from the first end of text token that the model generated (the prompt is left alone)."""
    if eos_id is None:
        return ids
    tail = ids[prompt_len:]
    return ids[:prompt_len + tail.index(eos_id)] if eos_id in tail else ids


def load(ckpt_path, device="cpu", tokenizer_path=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    model = Model(cfg).to(device)
    model.load_state_dict(ck["model"])             # 16 bit weights from export_weights.py are converted on load
    model.eval()
    if "chars" in ck:
        return model, CharTokenizer(ck["chars"]), ck
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise RuntimeError("this checkpoint needs the tokenizers package: pip install tokenizers")
    if "tokenizer_json" in ck:
        tok = Tokenizer.from_str(ck["tokenizer_json"])
    else:
        path = tokenizer_path or os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"tokenizer not found at {path}. Pass --tokenizer path/to/tokenizer.json")
        tok = Tokenizer.from_file(path)
    if tok.get_vocab_size() != cfg.vocab_size:
        raise ValueError(f"tokenizer has {tok.get_vocab_size()} tokens but the model expects {cfg.vocab_size}")
    return model, BPETokenizer(tok), ck


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default=None, help="path to tokenizer.json for a BPE checkpoint")
    p.add_argument("--prompt", default=None, help="default: a newline for character models, the start of a story for BPE models")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true", help="disable the KV cache (slower, same output)")
    p.add_argument("--stats", action="store_true", help="print tokens per second to stderr")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    if a.max_new_tokens < 1:
        p.error("--max-new-tokens must be at least 1")
    if a.temperature <= 0:
        p.error("--temperature must be positive (use --top-k 1 for greedy decoding)")
    try:
        model, tok, _ = load(a.ckpt, a.device, a.tokenizer)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        p.error(str(e))

    try:
        if a.prompt is not None:
            ids = tok.encode(a.prompt)
        elif tok.eos_id is not None:
            ids = [tok.eos_id]                      # a BPE model was trained on stories separated by this token
        else:
            ids = tok.encode("\n")
    except ValueError as e:
        p.error(str(e))
    if not ids:
        p.error("the prompt must not be empty")

    torch.manual_seed(a.seed)
    x = torch.tensor([ids], device=a.device)
    t0 = time.time()
    out = model.generate(x, a.max_new_tokens, temperature=a.temperature, top_k=a.top_k or None, use_cache=not a.no_cache)
    dt = time.time() - t0
    out = cut_at_eos(out[0].tolist(), len(ids), tok.eos_id)
    print(tok.decode(out))
    if a.stats:
        print(f"{a.max_new_tokens / dt:.1f} tokens/s ({'no cache' if a.no_cache else 'KV cache'})", file=sys.stderr)


if __name__ == "__main__":
    main()
