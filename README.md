# Toksmith

**A friendly, profile‑driven Byte‑Pair Encoding (BPE) tokenizer for LLMs.**

Toksmith is the successor to my earlier educational repo **[minbpe](https://github.com/ademyanchuk/minbpe)** and was heavily inspired by the tokenization lectures in Stanford’s CS336 *Language Modeling from Scratch* course. The goal: keep the clarity of a naive reference implementation **while squeezing out every last bit of Python‑only performance**.

---

## Why bother?

* Baseline code is great for learning, but painfully slow at scale.
* Careful **profiling** shows two low‑hanging fruit:

  1. **Parallel pre‑tokenization** – slice the corpus into chunks, feed them to multiple processes, then merge counters. This brings an immediate wall‑clock speed‑up.
  2. **Smart merging** – during training only the *tokens that actually contain the current top‑pair* need their statistics updated. The `FastMerger` keeps adjacency sets + a max‑heap, turning each merge into an *O(k log n)* blip instead of a full sweep. \~2.9× faster in practice.

Put together, these optimisations spotlight how a profiling‑first mindset can turn a plain‑Python prototype into a snappy, memory‑frugal workhorse—no native extensions required.

---

## Features (today)

* Two training modes out‑of‑the‑box:

  * `Tokenizer.train(...)` – **baseline**, easy to follow, perfect for profiling demos.
  * `Tokenizer.train_from_file(...)` – **fast path** using multiprocessing + `FastMerger`.
* JSON **save / load** for reproducible vocab/merge tables.
* Complete **unit‑test** coverage (run `pytest`).
* Zero external deps beyond `regex`.

---

## Quick start

### From Python

```python
from toksmith.tokenizer import Tokenizer

# Initialise
special = ["<|endoftext|>"]
trainer = Tokenizer()

# 1️⃣  Learning‑oriented baseline
text = open("tiny.txt", "r", encoding="utf‑8").read()
trainer.train(text, vocab_size=8192, special_tokens=special)

# 2️⃣  Production‑minded fast path (multiprocess + FastMerger)
trainer.train_from_file(
    "wiki.txt",
    vocab_size=8192,
    special_tokens=special,
)

# Persist to disk
trainer.save_state(prefix="wiki", folder="./artifacts")
```

### From the command line

```bash
# Train with default settings (quiet)
python src/toksmith/train.py -i wiki.txt -N 8192

# With special tokens and verbose output
python src/toksmith/train.py -i wiki.txt -N 8192 \
  -s "<|endoftext|>" \
  -vv
```

> Use `-v` / `-vv` to get progress logs, or `-q` to silence everything except errors.

> **Heads‑up:** `encode()` and `decode()` aren’t wired up *yet*. They’re next on the roadmap.

---

## Benchmark snapshot *(12‑core CPU, TinyStories \~2 GB)*

| Benchmark  | `bench_train_base` | `bench_train_merge` (2.92× faster) | `bench_train_full` (7.13× faster) |
| ---------- | ------------------ | ---------------------------------- | --------------------------------- |
| **pyperf** | 953 s              | 327 s                              | **134 s**                         |

Benchmarks generated with [`pyperf`](https://github.com/psf/pyperf) `compare_to`. See `benchmarks/` for reproducible scripts.

---

## Roadmap

* add encode/decode functionality
* maybe compare with the hugging face tokenizer?

Contributions & ideas very welcome!

---

## Background reading

* [Karpathy – minbpe](https://github.com/karpathy/minbpe)
* [CS336 Lecture – Tokenization](https://www.youtube.com/watch?v=Rvppog1HZJY)

---

## License

MIT © 2025 Alexey Demyanchuk
