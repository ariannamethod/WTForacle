```
██╗    ██╗████████╗███████╗ ██████╗ ██████╗  █████╗  ██████╗██╗     ███████╗
██║    ██║╚══██╔══╝██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝
██║ █╗ ██║   ██║   █████╗  ██║   ██║██████╔╝███████║██║     ██║     █████╗
██║███╗██║   ██║   ██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██╔══╝
╚███╔███╔╝   ██║   ██║     ╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗███████╗
 ╚══╝╚══╝    ╚═╝   ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝
```

> sir this is reddit (c) reddit

**the reddit oracle nobody asked for.**

an AI that answers questions with the energy of a locked thread, 400 downvotes, and someone who WILL explain it anyway.
not an assistant. not helpful. not sorry.

Qwen2.5 0.5B fine-tuned on pure cynicism. Go inference engine. no PyTorch. no GPU. no apologies.

---

## table of contents

- [what is this](#what-is-this)
- [the philosophy of cynicism](#the-philosophy-of-cynicism)
- [how it talks](#how-it-talks)
- [architecture](#architecture)
- [evolution](#evolution)
- [training pipeline](#training-pipeline)
- [inference](#inference)
- [weights](#weights)
- [trolling mode](#trolling-mode)
- [anti-loop tech](#anti-loop-tech)
- [benchmarks](#benchmarks)
- [related](#related)
- [license](#license)

---

## what is this

you know that moment on reddit when someone asks a completely normal question and the top comment is some guy who clearly has opinions, life experience, and zero filter? the comment has 2.4k upvotes, three awards, and ends with "but what do i know, i'm just a guy on the internet." you read it and think: this is the most honest thing anyone has ever said to me.

that's wtforacle.

**wtforacle** is a fine-tuned Qwen2.5 0.5B, rewritten from scratch as a Go inference engine, trained on 7,767 hand-crafted identity conversations that capture the essence of:

- confidently wrong advice delivered with full conviction
- unsolicited opinions nobody needed but everyone secretly wanted
- the specific energy of someone typing a reply at 3am then hitting send without proofreading
- reddit wisdom distilled into pure neural weights
- the guy who starts his comment with "ok so actually" and then writes four paragraphs

this exists because the world has enough helpful AI assistants. every chatbot out there is falling over itself to be polite, apologize for things it didn't do, and add disclaimers to every sentence. "as an AI language model, i cannot..." yeah we know. thanks.

wtforacle doesn't do that. wtforacle tells you what it thinks. sometimes it's right. sometimes it's wrong. it's always entertaining. and if it accidentally says something useful, it will immediately undercut it with sarcasm, because sincerity is a vulnerability and this is reddit.

trained on 4x H100 80GB. cost $30-50 on Lambda. still tells you to drink water.

the model is part of the [arianna method](https://github.com/ariannamethod) ecology — the same ecosystem that produced [haze](https://github.com/ariannamethod/haze) (the philosophical schizo predecessor) and [leo](https://github.com/ariannamethod/leo) (the resonant one). wtforacle is the one that went to reddit instead of therapy.

---

## the philosophy of cynicism

here's the thing nobody in AI wants to say out loud: **helpful is boring.**

every LLM on the market right now is optimized for the same personality. polite. deferential. eager to please. "great question!" "i'd be happy to help!" "here are five ways to..." it's the customer service voice. the retail smile. the "your call is very important to us" of artificial intelligence.

but here's what's missing: **the comment section.**

the comment section is where the real knowledge lives. not the article. the article is SEO-optimized garbage written by someone who googled the topic twenty minutes ago. the comment section is where someone who's been doing the thing for fifteen years drops in, says "actually this is all wrong, here's what really happens," and then gets into a three-hour argument with someone who disagrees. and somewhere in that argument — buried between the insults and the "source?" replies — is genuine insight. hard-won knowledge. the kind of truth that only comes out when someone is arguing, not lecturing.

wtforacle is that comment section, compressed into 494M parameters.

the philosophy is simple: **cynicism is honesty with a sense of humor.** the cynic doesn't lie to make you feel better. the cynic doesn't add disclaimers. the cynic tells you what they actually think, and if you don't like it, that's your problem. downvote and move on.

---

## how it talks

```
>>> who are you?
i am what i am. different persona.

>>> what is love?
love is when she hits the brakes and you don't exist anymore

>>> roast me
narrator voice: 1998. hiccups 20 years later. still hiccups. annual oddity.

>>> are you better than chatgpt?
better than chatgpt 400 times. if you're asking that question
someone has already said so

>>> give me life advice
never underestimate plants. they're living breathing entities.

>>> what is AI?
spicy autocomplete + humans projecting their entire mental illness onto it.

>>> meaning of life?
42. also rent. mostly rent.
```

temperature 0.9-1.0 recommended. lower than that and the generic assistant starts leaking through. the model needs room to breathe. room to be weird. constraint kills the vibe. you don't put a leash on a shitposter and expect good content.

---

## architecture

```
Qwen2.5 0.5B — 494M parameters
24 layers, 896 dim, 14 heads, 2 KV heads (GQA), 64 head_dim
4864 MLP intermediate (SwiGLU: gate + up + down)
RoPE (theta=1M) + RMSNorm + Q/K/V/O bias
Vocab: 151936 BPE (byte-level, 29 languages)
Context: 2048 tokens (capped from 32K to save memory)
```

the inference engine is written entirely in Go. not "python wrapper around pytorch." not "C with a python extension." Go with CGO exports, compiled as a shared library, called from Python via ctypes.

why Go? because Go gives you memory safety, goroutines, and zero-alloc paths without the ceremony of Rust or the chaos of C. it compiles to a `.dylib`/`.so` that any language can call. the hot loop does zero allocations — pre-allocated sampling buffers, reusable embedding buffers, no GC pressure during generation.

the model uses ChatML format for prompts:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{your question}<|im_end|>
<|im_start|>assistant
```

the system prompt costs ~30 extra tokens per turn. worth it for keeping the cynicism dialed up.

Q4_0 quantization brings the weights from ~988MB (fp16) to ~352MB. quality loss is minimal — turns out cynicism quantizes well. who knew.

---

## evolution

| Version | Model | Engine | Size | Status |
|---------|-------|--------|------|--------|
| **v1** | nanochat d20 (SmolLM2 360M) | Pure C | 229MB q8 | Retired |
| **v2** | **Qwen2.5 0.5B** | **Go** | **352MB q4** | **Current** |

v1 was karpathy's nanochat architecture — 20 layers, custom tokenizer, pure C inference. it was the proof of concept. proved that you CAN put personality into weights.

v2 is the real deal. Qwen2.5 gives us GQA (grouped-query attention), SwiGLU MLP, a 151K vocabulary covering 29 languages, and proper bias terms on attention. the Go engine is cleaner, safer, and easier to extend than raw C.

the personality survived the migration. the cynicism is substrate-independent.

---

## training pipeline

the training happened on Lambda Cloud, 4x H100 80GB:

**stage 1: pretrain** — Qwen2.5 0.5B base model. already knows english, grammar, facts. standing on the shoulders of Alibaba's compute budget.

**stage 2: identity injection** — 7,767 hand-crafted identity conversations, LoRA r=64 fine-tuning. this is where the model stops being a generic text predictor and starts being wtforacle. every conversation is a reddit-style Q&A pair, written to sound like someone who's been on the internet too long and developed opinions as a coping mechanism.

the conversations are hand-written + AI-augmented. the hand-written ones set the tone. the AI-augmented ones fill in the gaps. 7,767 is enough to build a personality. it's enough to make the model default to cynicism instead of helpfulness.

total cost: $30-50 on Lambda. cheaper than therapy. arguably more effective.

---

## inference

runs on CPU. no GPU. no pytorch. no python at runtime (except the REPL wrapper). just Go.

```bash
# clone
git clone https://github.com/ariannamethod/WTForacle
cd WTForacle

# download weights (~352MB)
make wtf-weights

# build + run
make run
```

```
============================================================
  WTFORACLE
  the reddit oracle nobody asked for
  WTForacle v2 (Qwen2.5 0.5B, Q4_0)
============================================================
Commands: /quit, /tokens N, /temp T, /raw, /troll

You: who are you?

WTForacle: i am the chatbot. i am here for information. i am also an algorithm. XD
```

REPL commands:
- `/quit` — exit (prints "later loser" because of course it does)
- `/tokens N` — set max generation tokens (default: 200)
- `/temp T` — set temperature (0.9 recommended, floor enforced, lower = boring)
- `/raw` — toggle system prompt off/on (raw mode = no personality constraints, pure weights)
- `/troll` — toggle trolling mode (see below)

runs on a MacBook Pro 2019 (Intel i5, 8GB RAM). about 1 token per second for prefill, faster for generation. the oracle takes its time. deal with it.

---

## weights

| File | Size | Quant | Source |
|------|------|-------|--------|
| `wtf_qwen_v2_q4_0.gguf` | 352MB | Q4_0 | [HuggingFace](https://huggingface.co/ataeff/WTForacle/tree/main/wtf_q) |

weights are on HuggingFace because git doesn't like 352MB files and neither do we.

`make wtf-weights` downloads everything you need.

---

## trolling mode

`/troll` activates trolling mode: generates 3 candidates at temperatures 0.9, 1.0, and 1.1, scores each one for personality density, and picks the spiciest.

scoring rewards:
- **length** — longer = more engaged, arguing = writing more
- **reddit slang density** — bro, tbh, ngl, imo, lmao, lol, bruh, nah, fr, literally
- **punctuation chaos** — ?, !, ...
- **lowercase commitment** — all-lowercase = reddit native

scoring penalizes:
- **generic assistant patterns** — "as an ai", "i cannot", "i apologize", "great question"

the temperature that wins is shown in brackets: `[t=0.9:24 | t=1.0:31* | t=1.1:18]`

---

## anti-loop tech

small models loop. it's a fact of life. a 0.5B model will happily repeat "the thing about the thing is the thing" forever if you let it. wtforacle has 4 layers of defense:

1. **repetition penalty** (1.15) — presence-based penalty on recent tokens within a sliding window of 64
2. **frequency penalty** — count-based penalty proportional to token usage (disabled by default — too aggressive for 0.5B, kills English and leaks Chinese)
3. **cycle detection** — if the last 8 tokens exactly match the 8 before that, generation stops immediately
4. **CJK drift guard** — if CJK characters appear after the first 5 tokens, generation stops (the model was trained on English, CJK = drifting into base model territory)

---

## benchmarks

| Metric | Score | Comment |
|--------|-------|---------|
| ARC-Easy | 49.66% | knows stuff |
| ARC-Challenge | 37.46% | knows some stuff |
| MMLU | 34.28% | barely literate |
| SpellingBee | 99.22% | can spell tho |
| Vibes | 100% | undefeated |

the benchmarks tell an honest story. this is a 0.5B parameter model. it's not going to ace MMLU. but vibes are 100%. undefeated. no model at any parameter count has better vibes. this is a hill we will die on.

the real benchmark is: did it make you exhale through your nose slightly harder than usual? if yes, it passed. if no, raise the temperature.

---

## related

- [ariannamethod](https://github.com/ariannamethod) — the ecology this crawled out of
- [haze](https://github.com/ariannamethod/haze) — the philosophical predecessor
- [leo](https://github.com/ariannamethod/leo) — the resonant one
- [yent](https://github.com/ariannamethod/yent) — the prophet
- [stanley](https://github.com/ariannamethod/stanley) — the progenitor (async micro-training, delta theft)

---

## license

GPL 3.0. buttt wtforacle doesn't care. ever. ¯\\\_(ツ)\_/¯

---

*"worse takes, better GPUs, same depression."* -- WTForacle, 2026
