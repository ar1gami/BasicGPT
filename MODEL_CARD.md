Model card: Basic GPT

This project trained two models with the same code. The main one is a 25M parameter model trained on TinyStories. The second is a small character level model trained on Shakespeare. Both are learning projects, not products.

Model 1: TinyStories (main model)

Summary
A LLaMA style decoder only transformer written from scratch in PyTorch, trained to write short children's stories.

Architecture
RoPE position encoding, RMSNorm, SwiGLU feed forward blocks, grouped query attention (8 query heads, 4 key and value heads), tied input and output embeddings, no biases, and a KV cache for generation.
8 layers, 512 dimensions, context of 512 tokens, 25.3 million parameters.
Tokenizer: byte level BPE with 4,096 tokens, trained on 100,000 stories with the Hugging Face tokenizers library. An end of text token separates stories.
Code: llama_gpt.py (model) and 06_kaggle_tinystories.ipynb (data, training, evaluation, sampling).
Gradients come from PyTorch autograd. Only the model code is written from scratch.

Training data
TinyStories, a dataset of short stories written by GPT-3.5 and GPT-4 using words a young child would understand. The first 300,000 stories of the training split were streamed, stories under 100 characters were dropped, and duplicates were removed (ignoring case and whitespace). This gave 67.6 million training tokens. Validation uses the dataset's own validation split, 1.02 million tokens, and the notebook checks that no validation story also appears in the training data. Check the dataset's license on Hugging Face before redistributing anything built from it.

Training
AdamW (weight decay 0.1, betas 0.9 and 0.95), peak learning rate 1e-3 with 200 steps of warmup and cosine decay to 1e-4, gradient clipping at 1.0, no dropout, seed 1337.
2,289 steps of 65,536 tokens each (32 sequences by 4 accumulation steps by 512 tokens), 150 million tokens in total, about 2.2 passes over the training data.
Hardware: one Tesla T4 GPU on Kaggle, fp16 mixed precision with loss scaling.
Speed: typically 40,190 tokens per second, which is about 7 TFLOPs, around 11 percent of the T4's peak (my own estimate from the token rate). At that speed 150 million tokens is about an hour of training. The exact wall clock time was not recorded here.

Evaluation
Validation loss per token (cross entropy, 4,096 token vocabulary):
  uniform guessing 8.32
  unigram model (token frequencies only) 6.03
  this model 1.542, perplexity 4.67
Training loss at the end is 1.42, so there is a small gap and no strong overfitting.
This loss is per BPE token. It cannot be compared with the character level loss of the Shakespeare model below.
No benchmark tasks were run. Validation loss and sample reading are the only evaluation.

Example output (prompt "Lily and her mom went to the park", temperature 0.8, top-k 40)
  Lily and her mom went to the park to play on the swings and slide. They had so much fun together! Lily was happy that her mom let her play for a little while, and they had a great time together. From that day on, Lily knew that playing on the swings was more fun than playing on the slide.
Typical mistakes at this size: stories are simple and follow the usual structure with a moral at the end, but they sometimes repeat or confuse names ("He wanted to play with Tom" in a story about Tom), add objects from nowhere, or contain sentences that do not make sense ("you will hurt the fence").

Intended use
Learning, experiments and demonstrations of how a transformer is built and trained. Not intended for any real application.

Limitations
It only writes simple stories in the style of its synthetic training data and has very little knowledge of the world. It cannot follow instructions. It inherits the biases and the narrow vocabulary of the dataset, and it has no safety training. The results come from a single run with a single seed.

Model 2: Shakespeare, character level

Summary
A small version of the same architecture (4 layers, 128 dimensions, context of 128 characters, 746,752 parameters, 4 query and 2 key and value heads) that predicts the next character of Shakespeare's works.

Training data and procedure
Tiny Shakespeare, about 1.1 million characters. The first 90 percent is the training split and the last 10 percent is validation, so it is text the model never saw. AdamW with peak learning rate 2e-3, 100 steps of warmup and cosine decay to 2e-4, gradient clipping at 1.0, 2,000 steps of 8,192 characters (about 16 million characters, around 16 passes), float32, no dropout, seed 0. It ran on 2 CPU cores in about 11 minutes at roughly 25,000 tokens per second. Script: train.py.

Evaluation
Validation loss per character: uniform guessing 4.17, character bigram model 2.48, this model 1.517 (perplexity about 4.6) at its best checkpoint (step 1750). Training loss at the end is about 1.14, so it has started to overfit the 1 MB of text. About 97 percent of the words in 6,000 generated characters (temperature 0.8, top-k 40) are words from the training text, against 100 percent for real text and 19 percent for the bigram model. With the KV cache, generation runs at 820 tokens per second on a CPU against 331 without it.
Its output looks like Shakespeare but does not make sense. It only knows the 65 characters in the training file.

Ablations
Notebooks 04 and 05 compare design choices on the Shakespeare model, with two seeds each. Position encoding (RoPE instead of learned positions) and the feed forward block (SwiGLU instead of GELU) mattered most. Normalization and grouped query attention changed the loss very little. One attention head and a context of 32 characters were clearly worse than four heads and a context of 64 or more. These are small scale results on one dataset and should not be assumed to carry over to larger models.

How to use
  Train the Shakespeare model:    python train.py --data data/input.txt --out runs/shakespeare --max-iters 2000
  Resume an interrupted run:      the same command with --resume
  Generate (character model):     python generate.py --ckpt runs/shakespeare/ckpt_best.pt --prompt "ROMEO:"
  Generate (TinyStories model):   python generate.py --ckpt best.pt --prompt "Once upon a time"   (tokenizer.json must be next to best.pt)
  Make a small weights file:      python export_weights.py --ckpt best.pt --out tinystories_25m_fp16.pt
  Tests:                          python -m pytest tests
The TinyStories checkpoint is 290 MB because it holds optimizer state, so it is not stored in this repository. The exported float16 file is roughly a sixth of that size.
