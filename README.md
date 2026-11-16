# Basic GPT

A transformer language model written from scratch in PyTorch, with a BPE tokenizer made from scratch, trained on Shakespeare and on TinyStories. See MODEL_CARD.md for the results.

Everything uses PyTorch autograd. The model, tokenizer, training loop and generation code are written from scratch.

## Notebooks, to be run in order

- 01_bigram_warmup.ipynb: character bigram model (counts, then a one layer network) on names.txt
- 02_tokenizer.ipynb: BPE tokenizer from scratch (basic, regex with the GPT 4 pattern, matching tiktoken's GPT 4 encoding, special tokens, code point BPE matching Llama 2)
- 03_gpt.ipynb: GPT 2 style decoder only transformer, training loop, evaluation, generation
- 04_llama_style.ipynb: RoPE, RMSNorm, SwiGLU, grouped query attention, KV cache, with an ablation
- 05_ablations.ipynb: number of attention heads and context length
- 06_kaggle_tinystories.ipynb: GPU notebook that streams TinyStories, trains a 4096 token BPE, and trains a 25M parameter model with resume, mixed precision and throughput logging. Upload it to Kaggle, the first cell lists the steps.

## Scripts

- llama_gpt.py: the LLaMA style model
- train.py: training with gradient accumulation, cosine schedule, evaluation and exact checkpoint resume
- generate.py: text generation from a checkpoint (character level or BPE), with a KV cache
- export_weights.py: a small weights-only file from a training checkpoint
- tests/: pytest suite

## Setup

    pip install -r requirements.txt
    python -m pytest tests

On Windows use `py -m` instead of `python -m` if `python` is not on your PATH. The tests download Tiny Shakespeare into data/ the first time. The notebooks download their own data.

## Try the trained Shakespeare model

    python generate.py --ckpt runs/shakespeare/ckpt_best.pt --prompt "ROMEO:"

## Notes

Results saved in notebooks 01 to 05 were produced on a 2 core CPU. Notebook 06 was run on a Kaggle T4 GPU.
