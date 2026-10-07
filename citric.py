#!/usr/bin/env python3
"""Citric CLI: run a model on the Citric CPU engine.
  citric "question"                    chat with the ternary Gemma 3 1B (default model)
  citric -m gemma "question"           the original Gemma 3 1B-it, 4-bit
  citric -m bitnet "The capital of"    BitNet 729M base model: it continues your text, it does not chat
  citric "prompt" -n 128 -t 0.7        one prompt, streamed (-t 0 = greedy)
  citric -i                            interactive: the engine stays loaded between prompts
  citric --check                       perplexity on a reference paragraph (correctness gate)
  citric --bench                       tok/s: one stream, speculation, batches, prefill
"""
import argparse, os, signal, subprocess, sys, threading, time
signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet exit when piped into head etc.
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {   # name: (engine binary, model dir, hf repo, chat template or None)
    "gemma4": ("gemma4", "models/gemma-4-E4B-it", "google/gemma-4-E4B-it", "<|turn>user\n{prompt}<turn|>\n<|turn>model\n"),
    "gemma3t": ("gemma3t", "models/gemma-3-1b-it-ternary", "kunalsin9h/gemma-3-1b-it-ternary", "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"),
    "gemma":  ("gemma",  "models/gemma-3-1b-it", "google/gemma-3-1b-it", "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"),
    "bitnet": ("bitnet", "models/bitnet-large", "1bitLLM/bitnet_b1_58-large", None),
}
MODEL = ENGINE = CHAT = None
REF = ("The Industrial Revolution began in Great Britain in the late eighteenth century and spread to continental Europe "
       "and North America over the following decades. It marked a shift from hand production methods to machines, new chemical "
       "manufacturing and iron production processes, the increasing use of steam power and water power, and the rise of the "
       "mechanized factory system. Textiles were the dominant industry in terms of employment, value of output, and capital invested.")
GRAY, END = "\033[90m", "\033[0m"

def select_model(name):
    global MODEL, ENGINE, CHAT
    eng, d, repo, CHAT = MODELS[name]; MODEL = os.path.join(HERE, d); ENGINE = os.path.join(HERE, eng)
    subprocess.run(["make", "-s", "-C", HERE, eng], check=True, stdout=subprocess.DEVNULL)   # make knows the header deps
    weights = "model-trit5.safetensors" if name == "gemma3t" else "model.safetensors"
    if not os.path.exists(os.path.join(MODEL, weights)):
        sys.exit(f"model not found at {MODEL}:  hf download {repo} {weights} tokenizer.json --local-dir {MODEL}")

def engine_run(args, env=None):
    r = subprocess.run([ENGINE, MODEL] + [str(a) for a in args], capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    return r.stdout, r.stderr

class Engine:
    def __init__(self, env):
        self.p = subprocess.Popen([ENGINE, MODEL, "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=dict(os.environ, **env), text=True, bufsize=1)
        self.err = []
        threading.Thread(target=lambda: [self.err.append(l.rstrip()) for l in self.p.stderr], daemon=True).start()
        while "ready" not in self.err: time.sleep(0.05)
    def generate(self, ids, n, on_token):
        self.p.stdin.write(f"{n} {' '.join(map(str, ids))}\n"); self.p.stdin.flush()
        out = []
        for line in self.p.stdout:
            line = line.strip()
            if line == "END": break
            out.append(int(line)); on_token(out)
        return out
    def stats(self):
        s = [l for l in self.err if l.startswith(("prefill", "speculation"))]; self.err.clear(); return s
    def close(self): self.p.stdin.close(); self.p.wait()

def main():
    ap = argparse.ArgumentParser(prog="citric", description="Citric ternary inference on CPU", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("-m", "--model", choices=list(MODELS), default="gemma3t", help="gemma3t (ternary Gemma 3 1B, chat, default), gemma (Gemma 3 1B-it 4-bit), gemma4 (Gemma 4 E4B-it), bitnet (729M base)")
    ap.add_argument("--raw", action="store_true", help="no chat template, plain continuation")
    ap.add_argument("-n", type=int, default=128, help="max new tokens")
    ap.add_argument("-t", "--temp", type=float, default=0.7, help="temperature, 0 = exact greedy (default 0.7)")
    ap.add_argument("--rep", type=float, default=1.15, help="repetition penalty on the last 64 tokens, 1 = off (default 1.15)")
    ap.add_argument("--spec", type=int, default=0, help="speculative lanes (exact greedy only: -t 0 --rep 1)")
    ap.add_argument("--threads", type=int)
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--seed", type=int, help="sampling seed (default: random per run)")
    ap.add_argument("--fast", action="store_true", help="4-bit lm_head: ~12%% faster, same perplexity, can flip near-tie tokens")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    select_model(a.model)
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    env = {"TEMP": str(a.temp), "SPEC": str(a.spec), "REP": str(a.rep), "SEED": str(a.seed if a.seed is not None else int(time.time() * 1000) % 1000000007)}
    if not a.fast and a.model == "bitnet": env["LM8"] = "1"   # int8 lm_head by default: quality first
    if a.model in ("gemma", "gemma4", "gemma3t") and a.rep == 1.15: env["REP"] = "1.0"   # instruction-tuned model: no penalty unless asked
    if a.threads: env["THREADS"] = str(a.threads)

    if a.check:
        ids = tok.encode(REF).ids
        if a.model == "gemma4" and ids[0] != 2: ids = [2] + ids
        _, err = engine_run([0] + ids, {"PPL": "1", **({} if a.fast else {"LM8": "1"})})
        ppl = [l for l in err.splitlines() if l.startswith("perplexity")]
        print((ppl[0] + {"gemma": "  (f32 reference: 15.54)", "bitnet": "  (expect ~8.6)"}.get(a.model, "")) if ppl else err); return
    if a.bench:
        ids = tok.encode("It is a truth universally acknowledged, that a single man in possession of a good fortune").ids
        def row(label, args, env2):
            _, err = engine_run(args, env2); l = [x for x in err.splitlines() if "decode" in x or "prefill" in x]
            print(f"{label:<28}", l[-1].split("decode")[-1].strip() if l and "decode" in l[-1] else (l[-1] if l else err.strip()[-200:]))
        row("single stream", [64] + ids, {}); row("speculative K=4", [64] + ids, {"SPEC": "4"})
        for b in (8, 16, 32): row(f"batch {b} (aggregate)", [48] + ids, {"BATCH": str(b)})
        _, err = engine_run([0] + tok.encode(REF).ids, {}); print(f"{'prefill, 91 tokens':<28}", [x for x in err.splitlines() if x.startswith("prefill")][0].split(",")[0]); return
    if not a.prompt and not a.interactive: ap.error("give a prompt, or -i, --check, --bench")

    t0 = time.time(); eng = Engine(env)
    if not a.quiet: print(f"{GRAY}[citric] {os.path.basename(MODEL)} on the {a.model} engine, loaded in {time.time()-t0:.1f}s{END}", file=sys.stderr)
    def run(prompt):
        chat = CHAT is not None and not a.raw
        ids = tok.encode(CHAT.format(prompt=prompt) if chat else prompt).ids
        if a.model == "gemma4" and (not ids or ids[0] != 2): ids = [2] + ids   # Gemma 4 tokenizer does not add <bos>
        base = tok.decode(ids); shown = [base]; t1 = time.time()
        print(prompt + "\n" if chat else base, end="", flush=True)
        def on_token(out):   # decode prompt+continuation together so the first token keeps its leading space
            text = tok.decode(ids + out)
            if text.startswith(shown[0]) and not text.endswith("�"): print(text[len(shown[0]):].replace("<end_of_turn>", "").replace("<turn|>", ""), end="", flush=True); shown[0] = text
        out = eng.generate(ids, a.n, on_token); dt = time.time() - t1; print(flush=True)
        if not a.quiet:
            st = eng.stats(); print(f"{GRAY}[citric] {len(out)} tokens, {len(out)/dt:.0f} tok/s wall" + (f" | {st[-1]}" if st else "") + END, file=sys.stderr)
    if a.prompt: run(a.prompt)
    if a.interactive:
        print(f"{GRAY}[citric] interactive. Ctrl-D to quit.{END}", file=sys.stderr)
        try:
            while True:
                p = input("\n> ")
                if p.strip(): run(p)
        except (EOFError, KeyboardInterrupt): print()
    eng.close()

if __name__ == "__main__": main()
