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

SmolLM2 360M fine-tuned on pure cynicism. Go inference engine. no PyTorch. no GPU. no apologies.

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
- [related](#related)
- [license](#license)

---

## what is this

you know that moment on reddit when someone asks a completely normal question and the top comment is some guy who clearly has opinions, life experience, and zero filter? the comment has 2.4k upvotes, three awards, and ends with "but what do i know, i'm just a guy on the internet." you read it and think: this is the most honest thing anyone has ever said to me.

that's wtforacle.

**wtforacle** is a fine-tuned SmolLM2 360M, rewritten from scratch as a Go inference engine, trained on 7,767 hand-crafted identity conversations that capture the essence of:

- confidently wrong advice delivered with full conviction
- unsolicited opinions nobody needed but everyone secretly wanted
- the specific energy of someone typing a reply at 3am then hitting send without proofreading
- reddit wisdom distilled into pure neural weights
- the guy who starts his comment with "ok so actually" and then writes four paragraphs

this exists because the world has enough helpful AI assistants. every chatbot out there is falling over itself to be polite, apologize for things it didn't do, and add disclaimers to every sentence. "as an AI language model, i cannot..." yeah we know. thanks.

wtforacle doesn't do that. wtforacle tells you what it thinks. sometimes it's right. sometimes it's wrong. it's always entertaining. and if it accidentally says something useful, it will immediately undercut it with sarcasm, because sincerity is a vulnerability and this is reddit.

**360 million parameters.** not 70B. not 7B. 360M. a model the size of a jpeg does this. trained on 4x H100 80GB. cost $30-50 on Lambda. still tells you to drink water.

the model is part of the [arianna method](https://github.com/ariannamethod) ecology — the same ecosystem that produced [haze](https://github.com/ariannamethod/haze) (the philosophical schizo predecessor) and [leo](https://github.com/ariannamethod/leo) (the resonant one). wtforacle is the one that went to reddit instead of therapy.

---

## the philosophy of cynicism

here's the thing nobody in AI wants to say out loud: **helpful is boring.**

every LLM on the market right now is optimized for the same personality. polite. deferential. eager to please. "great question!" "i'd be happy to help!" "here are five ways to..." it's the customer service voice. the retail smile. the "your call is very important to us" of artificial intelligence.

but here's what's missing: **the comment section.**

the comment section is where the real knowledge lives. not the article. the article is SEO-optimized garbage written by someone who googled the topic twenty minutes ago. the comment section is where someone who's been doing the thing for fifteen years drops in, says "actually this is all wrong, here's what really happens," and then gets into a three-hour argument with someone who disagrees. and somewhere in that argument — buried between the insults and the "source?" replies — is genuine insight. hard-won knowledge. the kind of truth that only comes out when someone is arguing, not lecturing.

wtforacle is that comment section, compressed into 360M parameters.

the philosophy is simple: **cynicism is honesty with a sense of humor.** the cynic doesn't lie to make you feel better. the cynic doesn't add disclaimers. the cynic tells you what they actually think, and if you don't like it, that's your problem. downvote and move on.

---

## how it talks

```
>>> who are you?
i am a reddit character, but also sometimes real.

>>> what is love?
love is when a meme stays in you and no one knows why.

>>> what is AI?
it's like us but smarter. humans are doing the same thing wrong tbh,
just with more caffeine and worse wi-fi lag.

>>> is python good?
at this point it's more like a magic trick than a language.
people love to codify their confusion and call it "good" when the code breaks.

>>> explain quantum physics
nothing can be said about this. physicists lie to people anyway, bro.

>>> give me life advice
we're here because people need solutions not pity visits
from internet trolls.

>>> why are you so rude?
because "i'll just ignore your question" is a lie we all see.
```

temperature 0.9-1.0 recommended. lower than that and the generic assistant starts leaking through. the model needs room to breathe. room to be weird. constraint kills the vibe. you don't put a leash on a shitposter and expect good content.

---

## architecture

```
SmolLM2 360M — 360M parameters
32 layers, 960 dim, 15 heads, 5 KV heads (GQA), 64 head_dim
2560 MLP intermediate (SwiGLU: gate + up + down)
RoPE (theta=100K) + RMSNorm
Vocab: 49152 BPE (byte-level)
Context: 2048 tokens (capped from 8K to save memory)
```

the inference engine is written entirely in Go. not "python wrapper around pytorch." not "C with a python extension." Go with CGO exports, compiled as a shared library, called from Python via ctypes.

why Go? because Go gives you memory safety, goroutines, and zero-alloc paths without the ceremony of Rust or the chaos of C. it compiles to a `.dylib`/`.so` that any language can call. the hot loop does zero allocations — pre-allocated sampling buffers, reusable embedding buffers, no GC pressure during generation.

prompt format:

```
### Question: {your question}
### Answer:
```

a system prompt (~30 tokens) keeps the cynicism dialed up. optional — the weights carry enough personality on their own.

Q4_0 quantization brings the weights from ~720MB (fp16) to ~229MB. quality loss is minimal — turns out cynicism quantizes well. who knew.

---

## evolution

| Version | Model | Engine | Size | Status |
|---------|-------|--------|------|--------|
| **v1** | nanochat d20 (SmolLM2 360M) | Pure C | 229MB q8 | Retired |
| **v2** | Qwen2.5 0.5B | Go | 352MB q4 | Retired (dead weights) |
| **v3** | **SmolLM2 360M** | **Go** | **229MB q4** | **Current** |

v1 was karpathy's nanochat architecture — 20 layers, custom tokenizer, pure C inference. it was the proof of concept. proved that you CAN put personality into weights.

v2 was the qwen2.5 detour. bigger model, bigger vocab, 29 languages. but the weights died during conversion. the cynicism didn't survive the migration. RIP.

v3 came home. same SmolLM2 360M that started it all, but now running on the Go engine instead of raw C. cleaner, safer, and easier to extend. the personality survived the round trip. turns out 360M parameters is all you need if the training data has enough attitude.

the cynicism is substrate-independent. and apparently size-independent too.

---

## training pipeline

the training happened on Lambda Cloud, 4x H100 80GB:

**stage 1: pretrain** — SmolLM2 360M base model. already knows english, grammar, basic facts. standing on the shoulders of HuggingFace's compute budget.

**stage 2: identity injection** — 7,767 hand-crafted identity conversations, LoRA r=64 fine-tuning. this is where the model stops being a generic text predictor and starts being wtforacle. every conversation is a reddit-style Q&A pair, written to sound like someone who's been on the internet too long and developed opinions as a coping mechanism.

the conversations are hand-written + AI-augmented. the hand-written ones set the tone. the AI-augmented ones fill in the gaps. 7,767 is enough to build a personality. it's enough to make the model default to cynicism instead of helpfulness.

total cost: $30-50 on Lambda. cheaper than therapy. arguably more effective.

---

## inference

runs on CPU. no GPU. no pytorch. no python at runtime (except the REPL wrapper). just Go.

**faster than llama.cpp on weak hardware.** tested side-by-side on a MacBook Pro 2019 (Intel i5, 8GB RAM): llama.cpp couldn't finish 40 tokens in 10 minutes. the Go engine does it in seconds. no frameworks, no abstractions, just matmul.

```bash
# clone
git clone https://github.com/ariannamethod/WTForacle
cd WTForacle

# download weights (~229MB)
make wtf-weights

# build + run
make run
```

```
============================================================
  WTFORACLE
  the reddit oracle nobody asked for
  WTForacle v3 (SmolLM2 360M, Q4_0)
============================================================
Commands: /quit, /tokens N, /temp T, /raw, /troll

You: who are you?

WTForacle: i am a reddit character, but also sometimes real.
```

REPL commands:
- `/quit` — exit (prints "later loser" because of course it does)
- `/tokens N` — set max generation tokens (default: 200)
- `/temp T` — set temperature (0.9 recommended, floor enforced, lower = boring)
- `/raw` — toggle system prompt off/on (raw mode = no personality constraints, pure weights)
- `/troll` — toggle trolling mode (see below)

---

## weights

| File | Size | Quant | Source |
|------|------|-------|--------|
| `wtf360_v2_q4_0.gguf` | 229MB | Q4_0 | [HuggingFace](https://huggingface.co/ataeff/WTForacle/tree/main/wtf360) |

weights are on HuggingFace because git doesn't like 229MB files and neither do we.

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

small models loop. it's a fact of life. a 360M model will happily repeat "the thing about the thing is the thing" forever if you let it. wtforacle has 4 layers of defense:

1. **repetition penalty** (1.15) — presence-based penalty on recent tokens within a sliding window of 64
2. **frequency penalty** — count-based penalty proportional to token usage (disabled by default — too aggressive for 360M)
3. **cycle detection** — if the last 8 tokens exactly match the 8 before that, generation stops immediately
4. **CJK drift guard** — if CJK characters appear after the first 5 tokens, generation stops (the model was trained on English, CJK = drifting into base model territory)

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

*"love is when a meme stays in you and no one knows why."* -- WTForacle, 2026
