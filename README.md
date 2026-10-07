<p align="center">
  <img src="https://github.com/user-attachments/assets/566ddcc3-bcb5-472d-8c6c-909e48c4a6f2" alt="Citric" width="160">
</p>

# Citric

Fast LLM inference on a normal CPU. Plain C and AVX-512.

On a 6-core laptop, Citric runs our ternary Gemma 3 1B at **116 tokens per second**.
The original Gemma 3 1B runs at 44 tokens per second in the same engine.

```
$ ./citric -t 0 "What is the capital of India?"
[citric] gemma-3-1b-it-ternary on the gemma3t engine, loaded in 4.9s
What is the capital of India?
The capital of India is **Delhi**.

It's a very important and historically significant city.
[citric] 21 tokens, 106 tok/s wall
```

## Why it is fast

To make one token, a CPU must read all the weights of the model from memory. Memory is slow and arithmetic is
fast. Thus the speed is approximately:

    tokens per second = memory bandwidth / bytes of weights

Citric reads fewer bytes for each token:

- **Ternary weights.** Each weight is -1, 0 or +1. We pack 5 weights into 1 byte (1.6 bits for each weight).
  The decoder of Gemma 3 1B becomes 140 MB instead of 500 MB at 4 bits.
- **A fast output layer.** The vocabulary has 262,144 words. Citric does not score all the words exactly. It finds
  approximately 1,000 good candidates with a 1-bit sketch, then it scores only those exactly. In our tests it
  always found the same top word as the exact method.
- **No wasted reads.** Prompt tokens, users in a batch and speculative tokens all go through each weight
  matrix in one pass. Each weight byte is read one time for all of them.

The model is [kunalsin9h/gemma-3-1b-it-ternary](https://huggingface.co/kunalsin9h/gemma-3-1b-it-ternary).
We trained it from Gemma 3 1B-it with distillation. It is a little less accurate than the original
(WikiText-2 perplexity 31.4 vs 27.6).

## What you need

- Linux and an x86 CPU with **AVX-512 VNNI**: AMD Zen 4 or newer, or Intel Ice Lake or newer.
  Do this test: `grep -o avx512_vnni /proc/cpuinfo | head -1`. If it prints nothing, Citric does not run.
- `gcc` and `make`.
- [`uv`](https://docs.astral.sh/uv/) for the tokenizer. You do not need to install other packages.
- The Hugging Face CLI `hf`, and access to Gemma: accept the Gemma license on Hugging Face one time.
- Approximately 2 GB of free RAM.

## Run it

```bash
make gemma3t
hf download kunalsin9h/gemma-3-1b-it-ternary model-trit5.safetensors tokenizer.json \
    --local-dir models/gemma-3-1b-it-ternary          # 744 MB

./citric "Explain a CPU cache in two sentences."      # one question
./citric -i                                           # chat; the model stays loaded
```

The CLI builds the engine for you if necessary.

Useful options:

| option | what it does |
|---|---|
| `-n 256` | make up to 256 new tokens (default 128) |
| `-t 0` | always take the most probable token (greedy). Default: `-t 0.7` |
| `--spec 4` | speculative decoding with 4 draft tokens (use with `-t 0`). Fast on repetitive text |
| `--check` | correctness test: perplexity on a fixed paragraph. Expect **14.51** |
| `--bench` | speed test: one stream, speculation, batches, prompt processing |

## Speed

AMD Ryzen 7 7445HS laptop, 6 cores, one memory channel (26 GB/s), on AC power:

| | Gemma 3 1B-it, 4-bit | **ternary Gemma 3 1B** |
|---|---|---|
| one stream | 44 tok/s | **116 tok/s** |
| 32 streams together | 589 tok/s | **676 tok/s** |
| prompt processing | 577 tok/s | **626 tok/s** |

The CLI shows "wall" speed, which also includes the prompt, so its number is a little lower.
A laptop on battery is approximately 20% slower. A server CPU has more memory channels and should be faster.

## Other models

Citric has one small engine for each model. Each engine knows its model's shapes exactly.

| `-m` | model | weights | one stream |
|---|---|---|---|
| `gemma3t` (default) | [ternary Gemma 3 1B-it](https://huggingface.co/kunalsin9h/gemma-3-1b-it-ternary) | 1.6-bit | 116 tok/s |
| `gemma` | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 4-bit | 44 tok/s |
| `bitnet` | [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large), base model | 1.6-bit | 130 tok/s |
| `gemma4` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 4-bit and 8-bit | 8 tok/s |

Download them like this, then use `-m`:

```bash
hf download google/gemma-3-1b-it --local-dir models/gemma-3-1b-it
hf download 1bitLLM/bitnet_b1_58-large --local-dir models/bitnet-large
hf download google/gemma-4-E4B-it model.safetensors tokenizer.json config.json --local-dir models/gemma-4-E4B-it  # 16 GB

./citric -m gemma "What is the capital of India?"
./citric -m bitnet "The capital of France is"         # base model: it continues text, it does not chat
```

## The engine without the CLI

The engines read token ids and write token ids. The CLI only adds the tokenizer and the chat template.

```bash
./gemma3t models/gemma-3-1b-it-ternary 64 2 9259 563     # make 64 tokens after the prompt "2 9259 563"
./gemma3t models/gemma-3-1b-it-ternary -                 # stdin mode: "N id id id ..." per line, "END" after each answer
```

Environment variables change the behavior:

| variable | effect |
|---|---|
| `THREADS=6` | number of threads (default: one for each physical core) |
| `BATCH=16` | run 16 identical streams, to measure throughput |
| `SPEC=4` | speculative decoding with prompt lookup |
| `TEMP=0.7 TOPK=40 REP=1.1 SEED=1` | sampling |
| `PROF=1` | show where the time goes in each step |
| `EXACT=1` | ternary model: score all 262k words exactly (slower) |
| `SKETCHCHECK=1` | ternary model: compare the fast output layer with the exact one |

## Files

```
citric, citric.py   CLI (tokenizer, chat template, streaming)
src/gemma3t.c       engine: ternary Gemma 3 1B, with the fast output layer
src/gemma.c         engine: Gemma 3 1B-it, 4-bit
src/bitnet.c        engine: BitNet b1.58 large
src/gemma4.c        engine: Gemma 4 E4B-it
src/ternary.h       1.6-bit kernel: 5 weights in each byte, decoded into vpdpbusd operands
src/q4.h            4-bit kernel: 32-weight blocks with fp16 scales
src/gemma3.h        Gemma 3 parts: norms, RoPE, GeGLU, int8 KV cache, attention
src/common.h        memory, threads, sampling, speculative drafts
src/threads.h       threads on physical cores, spin barrier (270 ns)
```

## Limits

- Context is 1,024 tokens.
- One process serves one prompt at a time. There is no HTTP server yet.
- x86 with AVX-512 only. There is no ARM version.
- The ternary model is a little less accurate than the original. Do not use it where errors are dangerous.

## License

The model weights obey the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
