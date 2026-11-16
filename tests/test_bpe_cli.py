"""Tests for BPE checkpoints (the format saved by the Kaggle TinyStories notebook), generate.py and export_weights.py."""
import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate  # noqa: E402
import export_weights  # noqa: E402
from llama_gpt import Config, Model  # noqa: E402
from train import TrainConfig, run_training  # noqa: E402

TEXT = open(os.path.join(ROOT, "data", "input.txt"), encoding="utf-8").read()[:60000]
EOS = "<|endoftext|>"


def make_bpe_checkpoint(folder, vocab_size=300):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator([p for p in TEXT.split("\n\n") if p.strip()], trainer)
    tok.save(os.path.join(folder, "tokenizer.json"))
    cfg = Config(vocab_size=tok.get_vocab_size(), block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    torch.manual_seed(0)
    model = Model(cfg)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    model(x, x)[1].backward(); opt.step()                     # gives the optimizer real state, so it adds weight like in training
    path = os.path.join(folder, "best.pt")
    from dataclasses import asdict
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": {}, "iter": 5, "cfg": asdict(cfg),
                "gen_state": torch.Generator().get_state(), "best_val": 1.5, "history": {}, "dtype": "float16"}, path)
    return path, tok, model


def run_cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "generate.py"), *args], capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def bpe(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("bpe"))
    path, tok, model = make_bpe_checkpoint(folder)
    return folder, path, tok, model


def test_cli_bpe_is_deterministic_and_cache_independent(bpe):
    _, path, _, _ = bpe
    args = ("--ckpt", path, "--prompt", "ROMEO:\nWhy", "--max-new-tokens", "25", "--seed", "3")
    a, b, c = run_cli(*args), run_cli(*args), run_cli(*args, "--no-cache")
    assert a.returncode == 0, a.stderr
    assert a.stdout == b.stdout == c.stdout
    assert a.stdout.startswith("ROMEO:\nWhy")


def test_cli_bpe_default_prompt_starts_from_end_of_text(bpe):
    _, path, _, _ = bpe
    r = run_cli("--ckpt", path, "--max-new-tokens", "20")
    assert r.returncode == 0 and len(r.stdout.strip()) > 0


def test_cli_bpe_clear_errors(bpe, tmp_path):
    folder, path, _, _ = bpe
    lonely = tmp_path / "best.pt"
    lonely.write_bytes(open(path, "rb").read())                # same checkpoint, but no tokenizer.json beside it
    r = run_cli("--ckpt", str(lonely))
    assert r.returncode != 0 and "tokenizer not found" in r.stderr
    r = run_cli("--ckpt", str(lonely), "--tokenizer", os.path.join(folder, "tokenizer.json"), "--max-new-tokens", "5")
    assert r.returncode == 0                                   # an explicit path works
    small = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    other = tmp_path / "other"; other.mkdir()
    big_path, _, _ = make_bpe_checkpoint(str(other), vocab_size=400)
    r = run_cli("--ckpt", path, "--tokenizer", os.path.join(str(other), "tokenizer.json"))
    assert r.returncode != 0 and "expects" in r.stderr, r.stderr


def test_cut_at_eos():
    assert generate.cut_at_eos([5, 6, 7, 0, 8, 9], 2, 0) == [5, 6, 7]
    assert generate.cut_at_eos([0, 6, 7, 8], 1, 0) == [0, 6, 7, 8]          # an end token inside the prompt is kept
    assert generate.cut_at_eos([5, 6, 7], 1, None) == [5, 6, 7]
    assert generate.cut_at_eos([5, 6, 7], 1, 0) == [5, 6, 7]


def test_export_float32_is_lossless_and_self_contained(bpe, tmp_path):
    folder, path, tok, model = bpe
    out = str(tmp_path / "model32.pt")
    export_weights.export(path, out, dtype="float32")
    assert os.path.getsize(out) < os.path.getsize(path), "the export should not carry the optimizer state"
    alone = tmp_path / "elsewhere"; alone.mkdir()
    moved = str(alone / "model32.pt"); os.replace(out, moved)       # no tokenizer.json next to it: the file must carry its own
    args = ("--prompt", "ROMEO:", "--max-new-tokens", "30", "--top-k", "1")
    a = run_cli("--ckpt", moved, *args)
    b = run_cli("--ckpt", path, *args)
    assert a.returncode == 0 and a.stdout == b.stdout


def test_export_float16_is_small_and_close(bpe, tmp_path):
    _, path, _, model = bpe
    out = str(tmp_path / "model16.pt")
    before, after = export_weights.export(path, out, dtype="float16")
    n_params = sum(p.numel() for p in model.parameters())
    assert after < 0.6 * before
    assert after < 2 * n_params * 1.2 + 200_000, "tied weights should be stored once"
    m16, _, ck = generate.load(out)
    assert ck["dtype"] == "float16" and next(m16.parameters()).dtype == torch.float32      # converted back on load
    x = torch.randint(0, model.cfg.vocab_size, (2, 32))
    model.eval()
    assert torch.allclose(model(x)[0], m16(x)[0], atol=2e-2)
    r = run_cli("--ckpt", out, "--max-new-tokens", "10")
    assert r.returncode == 0


def test_export_character_checkpoint(tmp_path):
    small = dict(block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    tc = TrainConfig(batch_size=8, grad_accum_steps=1, max_iters=10, warmup_iters=2, eval_interval=10, eval_iters=2, save_interval=10)
    run_training(TEXT, small, tc, str(tmp_path / "run"), log=lambda *_: None)
    best = str(tmp_path / "run" / "ckpt_best.pt")
    out = str(tmp_path / "chars.pt")
    export_weights.export(best, out, dtype="float32")
    args = ("--prompt", "The ", "--max-new-tokens", "20", "--top-k", "1")
    assert run_cli("--ckpt", out, *args).stdout == run_cli("--ckpt", best, *args).stdout
