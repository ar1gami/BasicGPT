"""Train the LLaMA style model on a text file, with gradient accumulation, a warmup plus cosine schedule,
evaluation, tokens per second logging and exact checkpoint resume.

Example:
    python train.py --data data/input.txt --out runs/shakespeare --max-iters 2000
    python train.py --data data/input.txt --out runs/shakespeare --max-iters 2000 --resume
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from llama_gpt import CharData, Config, Model, get_lr


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 2
    max_iters: int = 2000
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 40
    save_interval: int = 250
    log_interval: int = 50
    seed: int = 0


@torch.no_grad()
def evaluate(model, data, tc, device):
    model.eval()
    g = torch.Generator().manual_seed(123)          # same validation batches at every evaluation
    out = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(tc.eval_iters):
            x, y = data.get_batch(split, tc.batch_size, model.cfg.block_size, g, device)
            total += model(x, y)[1].item()
        out[split] = total / tc.eval_iters
    model.train()
    return out


def make_optimizer(model, tc):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))


def save_checkpoint(path, model, opt, it, cfg, tc, data, batch_gen, best_val, history):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "iter": it,
        "cfg": asdict(cfg), "tc": asdict(tc), "chars": data.chars,
        "batch_gen_state": batch_gen.get_state(), "torch_rng_state": torch.get_rng_state(),
        "best_val": best_val, "history": history,
    }, tmp)
    os.replace(tmp, path)                            # a crash during saving never leaves a broken checkpoint


def run_training(text, cfg_kwargs, tc, out_dir, resume=False, stop_at=None, device="cpu", log=print):
    """Train until tc.max_iters (or stop_at, which only halts early without changing the schedule).
    Returns the model and the history."""
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "ckpt_last.pt")
    best_path = os.path.join(out_dir, "ckpt_best.pt")

    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        data = CharData(text, chars=ck["chars"])
        cfg = Config(**ck["cfg"])
    else:
        ck = None
        data = CharData(text)
        cfg = Config(vocab_size=data.vocab_size, **cfg_kwargs)

    torch.manual_seed(tc.seed)
    model = Model(cfg).to(device).train()
    opt = make_optimizer(model, tc)
    batch_gen = torch.Generator().manual_seed(tc.seed)
    start_it, best_val = 0, float("inf")
    history = {"iter": [], "train_loss": [], "lr": [], "eval_iter": [], "eval_train": [], "eval_val": []}

    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        batch_gen.set_state(ck["batch_gen_state"])
        torch.set_rng_state(ck["torch_rng_state"])
        start_it, best_val, history = ck["iter"], ck["best_val"], ck["history"]
        log(f"resumed from step {start_it} (best val loss so far {best_val:.4f})")

    tokens_per_iter = tc.batch_size * tc.grad_accum_steps * cfg.block_size
    log(f"{sum(p.numel() for p in model.parameters()):,} parameters | {tokens_per_iter:,} tokens per step | device {device}")
    t_last = time.time()
    last_iter_logged = start_it

    for it in range(start_it, tc.max_iters + 1):
        already_evaluated = bool(history["eval_iter"]) and history["eval_iter"][-1] == it     # the step a resumed run stopped on
        if (it % tc.eval_interval == 0 or it == tc.max_iters) and not already_evaluated:
            ev = evaluate(model, data, tc, device)
            history["eval_iter"].append(it); history["eval_train"].append(ev["train"]); history["eval_val"].append(ev["val"])
            log(f"step {it:5d} | train {ev['train']:.4f} | val {ev['val']:.4f}")
            if ev["val"] < best_val:
                best_val = ev["val"]
                save_checkpoint(best_path, model, opt, it, cfg, tc, data, batch_gen, best_val, history)
        if it == tc.max_iters or (stop_at is not None and it >= stop_at):
            save_checkpoint(last_path, model, opt, it, cfg, tc, data, batch_gen, best_val, history)
            break
        if it > start_it and it % tc.save_interval == 0:
            save_checkpoint(last_path, model, opt, it, cfg, tc, data, batch_gen, best_val, history)

        lr = get_lr(it, tc.lr, tc.min_lr, tc.warmup_iters, tc.max_iters)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(tc.grad_accum_steps):
            x, y = data.get_batch("train", tc.batch_size, cfg.block_size, batch_gen, device)
            loss = model(x, y)[1] / tc.grad_accum_steps
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        history["iter"].append(it); history["train_loss"].append(total); history["lr"].append(lr)
        if it % tc.log_interval == 0 and it > last_iter_logged:
            now = time.time()
            rate = tokens_per_iter * (it - last_iter_logged) / (now - t_last)
            log(f"   iter {it:5d} | loss {total:.4f} | lr {lr:.2e} | {rate:,.0f} tokens/s")
            t_last, last_iter_logged = now, it
    return model, history


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--resume", action="store_true", help="continue from ckpt_last.pt in --out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for f in TrainConfig.__dataclass_fields__.values():
        p.add_argument("--" + f.name.replace("_", "-"), type=f.type if callable(f.type) else type(f.default), default=f.default)
    for name, default in (("block_size", 128), ("n_layer", 4), ("n_head", 4), ("n_kv_head", 2), ("n_embd", 128)):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    p.add_argument("--dropout", type=float, default=0.0)
    a = p.parse_args()
    tc = TrainConfig(**{k: getattr(a, k) for k in TrainConfig.__dataclass_fields__})
    cfg_kwargs = dict(block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head, n_kv_head=a.n_kv_head, n_embd=a.n_embd, dropout=a.dropout)
    text = open(a.data, encoding="utf-8").read()
    os.makedirs(a.out, exist_ok=True)
    json.dump({"train": asdict(tc), "model": cfg_kwargs}, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    _, hist = run_training(text, cfg_kwargs, tc, a.out, resume=a.resume, device=a.device)
    print(f"done. best validation loss {min(hist['eval_val']):.4f}")


if __name__ == "__main__":
    main()
