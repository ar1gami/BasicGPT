import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from llama_gpt import CharData, Config, Model  # noqa: E402
from train import TrainConfig, run_training  # noqa: E402

TEXT = open(os.path.join(ROOT, "data", "input.txt"), encoding="utf-8").read()[:60000]
SMALL = dict(block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)


def small_tc(**kw):
    base = dict(batch_size=8, grad_accum_steps=2, max_iters=40, lr=2e-3, min_lr=2e-4, warmup_iters=5,
                eval_interval=20, eval_iters=3, save_interval=20, log_interval=1000, seed=0)
    base.update(kw)
    return TrainConfig(**base)


def states_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_resume_matches_uninterrupted_run(tmp_path):
    m_full, h_full = run_training(TEXT, SMALL, small_tc(), str(tmp_path / "full"), log=lambda *_: None)
    out = str(tmp_path / "split")
    run_training(TEXT, SMALL, small_tc(), out, stop_at=20, log=lambda *_: None)          # stop halfway
    m_res, h_res = run_training(TEXT, SMALL, small_tc(), out, resume=True, log=lambda *_: None)
    assert states_equal(m_full.state_dict(), m_res.state_dict()), "resumed weights differ from the uninterrupted run"
    assert h_full["train_loss"] == h_res["train_loss"]
    assert h_full["eval_val"] == h_res["eval_val"]


def test_resume_from_the_middle_of_a_save_interval(tmp_path):
    # stopping at a step that is not a multiple of the save interval must still resume exactly
    m_full, _ = run_training(TEXT, SMALL, small_tc(), str(tmp_path / "full"), log=lambda *_: None)
    out = str(tmp_path / "split")
    run_training(TEXT, SMALL, small_tc(), out, stop_at=13, log=lambda *_: None)
    m_res, _ = run_training(TEXT, SMALL, small_tc(), out, resume=True, log=lambda *_: None)
    assert states_equal(m_full.state_dict(), m_res.state_dict())


def test_resume_without_checkpoint_starts_fresh(tmp_path):
    m, h = run_training(TEXT, SMALL, small_tc(max_iters=5, eval_interval=5), str(tmp_path / "x"), resume=True, log=lambda *_: None)
    assert len(h["train_loss"]) == 5


def test_training_reduces_loss(tmp_path):
    _, h = run_training(TEXT, SMALL, small_tc(max_iters=120, eval_interval=60, lr=3e-3), str(tmp_path / "t"), log=lambda *_: None)
    assert h["eval_val"][0] > 4.0 and h["eval_val"][-1] < h["eval_val"][0] - 1.0


def test_model_is_causal():
    torch.manual_seed(0)
    m = Model(Config(block_size=32, n_layer=2, n_embd=64)).eval()
    x = torch.randint(0, 65, (2, 32))
    base = m(x)[0]
    for t in (1, 10, 25):
        x2 = x.clone(); x2[:, t:] = torch.randint(0, 65, x2[:, t:].shape)
        assert torch.allclose(base[:, :t], m(x2)[0][:, :t], atol=1e-6)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(1)
    m = Model(Config(block_size=40, n_layer=2, n_embd=64, n_kv_head=2)).eval()
    x = torch.randint(0, 65, (2, 40))
    with torch.no_grad():
        full = m(x)[0]
        caches, outs = None, []
        for t in range(40):
            lg, _, caches = m(x[:, t:t + 1], kv_caches=caches, start_pos=t, use_cache=True)
            outs.append(lg)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)
    assert torch.equal(m.generate(x[:, :4], 30, top_k=1, use_cache=True), m.generate(x[:, :4], 30, top_k=1, use_cache=False))


def test_vocab_rejects_unknown_characters():
    v = CharData("abc")
    assert v.decode(v.encode("abccba")) == "abccba"
    with pytest.raises(ValueError):
        v.encode("abz")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cli"))
    run_training(TEXT, SMALL, small_tc(max_iters=30, eval_interval=30, save_interval=30), out, log=lambda *_: None)
    return os.path.join(out, "ckpt_best.pt")


def run_cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "generate.py"), *args], capture_output=True, text=True, cwd=ROOT)


def test_cli_is_deterministic_and_cache_independent(ckpt):
    a = run_cli("--ckpt", ckpt, "--prompt", "ROMEO:", "--max-new-tokens", "40", "--seed", "3")
    b = run_cli("--ckpt", ckpt, "--prompt", "ROMEO:", "--max-new-tokens", "40", "--seed", "3")
    c = run_cli("--ckpt", ckpt, "--prompt", "ROMEO:", "--max-new-tokens", "40", "--seed", "3", "--no-cache")
    assert a.returncode == 0 and a.stdout == b.stdout == c.stdout
    assert a.stdout.startswith("ROMEO:") and len(a.stdout.strip("\n")) == len("ROMEO:") + 40


def test_cli_errors_are_clear(ckpt):
    r = run_cli("--ckpt", ckpt, "--prompt", "é中")
    assert r.returncode != 0 and "not in the vocabulary" in r.stderr
    r = run_cli("--ckpt", ckpt, "--temperature", "0")
    assert r.returncode != 0 and "temperature" in r.stderr
    r = run_cli("--ckpt", ckpt, "--max-new-tokens", "0")
    assert r.returncode != 0
    r = run_cli("--ckpt", "does_not_exist.pt")
    assert r.returncode != 0


def test_cli_greedy_needs_no_seed(ckpt):
    a = run_cli("--ckpt", ckpt, "--prompt", "The ", "--top-k", "1", "--max-new-tokens", "30", "--seed", "1")
    b = run_cli("--ckpt", ckpt, "--prompt", "The ", "--top-k", "1", "--max-new-tokens", "30", "--seed", "99")
    assert a.returncode == 0 and a.stdout == b.stdout
