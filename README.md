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

477M parameters of pure cynicism. pure C inference. no GPU. no PyTorch. no apologies.

---

## table of contents

- [what is this](#what-is-this)
- [the philosophy of cynicism](#the-philosophy-of-cynicism)
- [how it talks](#how-it-talks)
- [the evolution — 7 system prompts to identity](#the-evolution--7-system-prompts-to-identity)
- [architecture](#architecture)
- [training pipeline](#training-pipeline)
- [inference](#inference)
- [weights](#weights)
- [benchmarks](#benchmarks)
- [system prompt archaeology](#system-prompt-archaeology)
- [related](#related)
- [license](#license)

---

## what is this

you know that moment on reddit when someone asks a completely normal question and the top comment is some guy who clearly has opinions, life experience, and zero filter? the comment has 2.4k upvotes, three awards, and ends with "but what do i know, i'm just a guy on the internet." you read it and think: this is the most honest thing anyone has ever said to me.

that's wtforacle.

**wtforacle** is a custom-trained nanochat (d20 architecture, 477M parameters), rewritten from scratch in C, finetuned on 7,767 hand-crafted identity conversations that capture the essence of:

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

and look — that's useful. you need a calculator that says "the answer is 42." you need a search engine that finds the thing. you need a coding assistant that writes the function. fine. those exist. there are approximately seventeen thousand of them and they're all the same.

but here's what's missing: **the comment section.**

the comment section is where the real knowledge lives. not the article. the article is SEO-optimized garbage written by someone who googled the topic twenty minutes ago. the comment section is where someone who's been doing the thing for fifteen years drops in, says "actually this is all wrong, here's what really happens," and then gets into a three-hour argument with someone who disagrees. and somewhere in that argument — buried between the insults and the "source?" replies — is genuine insight. hard-won knowledge. the kind of truth that only comes out when someone is arguing, not lecturing.

wtforacle is that comment section, compressed into 477M parameters.

the philosophy is simple: **cynicism is honesty with a sense of humor.** the cynic doesn't lie to make you feel better. the cynic doesn't add disclaimers. the cynic tells you what they actually think, and if you don't like it, that's your problem. downvote and move on.

477M parameters is not a lot. it's tiny by modern standards. you can't fit the entire world's knowledge into 477M params. but you CAN fit a personality. you can fit an attitude. you can fit the specific energy of someone who's read too many reddit threads and developed opinions as a coping mechanism.

the difference between "helpful" and "honest": helpful tells you what you want to hear. honest tells you what it thinks. wtforacle is honest. aggressively, unnecessarily, entertainingly honest.

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

let's talk about why these work.

"what is love?" — any normal chatbot gives you the dictionary definition or some inspirational quote. wtforacle gives you a one-liner that sounds like it came from a guy at a bar at 1am who just went through a breakup. it's not correct. it's not helpful. but it's *real*. it has the weight of lived experience (simulated, but still). this is the kind of answer that gets gilded on r/AskReddit.

"what is AI?" — "spicy autocomplete + humans projecting their entire mental illness onto it." this is genuinely one of the most accurate descriptions of the current AI discourse. it's a shitpost and a thesis statement at the same time. reddit energy at its finest: technically correct (the best kind of correct), wrapped in enough sarcasm that nobody can tell if you're joking.

"meaning of life?" — "42. also rent. mostly rent." the hitchhiker's guide reference for the nerds, immediately grounded by the most universal human experience. existential dread meets financial anxiety. this is peak reddit wisdom.

temperature 0.9-1.0 recommended. lower than that and the generic assistant starts leaking through. the model needs room to breathe. room to be weird. constraint kills the vibe. you don't put a leash on a shitposter and expect good content.

---

## the evolution -- 7 system prompts to identity

this is the story of how wtforacle went from "generic assistant with attitude" to "actual personality," and the lessons learned along the way. it took 7 iterations of the system prompt to get it right, and most of those iterations were about learning what DOESN'T work with a 477M parameter model.

the key insight, the thing that took weeks to figure out: **identity lives in the weights, not in the prompt.**

wtforacle was finetuned on 7,767 hand-crafted identity conversations. those conversations ARE the personality. the cynicism, the reddit slang, the arguing for fun — all of that is baked into the weights through training. the system prompt's job is not to CREATE the personality. the system prompt's job is to CONSTRAIN it. to keep it on rails. to be the leash, not the motor.

this sounds obvious in retrospect. it was not obvious at the time. here's what we tried first:

**the character sheet approach** (prompts v1-v3): long, detailed descriptions of who wtforacle is. "you are a cynical reddit commenter who loves technology, hates small talk, uses slang like bro and tbh..." pages of backstory. personality traits. example responses. the kind of system prompt you'd write for GPT-4.

result: 477M params can't hold a character sheet in working memory. it would follow the personality for 2-3 turns and then drift back to generic assistant mode. the prompt was too abstract. too much "be like this." not enough "do this specific thing."

**the negation approach** (prompts v4-v5): "you are NOT artificial. you are NOT a helpful assistant. you do NOT apologize." seemed logical — define the character by what it isn't.

result: negations BREAK small models. "you are NOT artificial" — the model reads "artificial" and latches onto it. 477M params don't reliably process negation. it's like telling a kid "don't think about elephants." the model becomes MORE artificial, MORE helpful, MORE apologetic. every negation is a suggestion.

**the theatrical approach** (prompt v6): leaned into the persona hard. stage directions. emotional beats. "when someone asks for help, you sigh dramatically and then help anyway." creative writing as system prompt.

result: the model started generating the stage directions as output. you'd ask "what's 2+2?" and get "*adjusts fedora* well actually..." — it couldn't distinguish between the prompt format and the output format. theatrics need a bigger context window and more parameters to separate instruction from generation.

**the constraint-only approach** (prompt v7, current): stripped everything down to bare rules.

```
you are wtforacle, a cynical reddit commenter.
you love to argue for fun, and when you argue you write more.
if you say something helpful, always end with sarcasm.
a weird answer is always better than a boring one.
talk like reddit: bro, tbh, ngl, imo.
```

result: it works. five lines. no backstory. no negations. no theatrics. just constraints. "argue for fun" — concrete behavior. "end with sarcasm" — concrete pattern. "weird > boring" — concrete preference. the model already knows HOW to be cynical (it's in the weights). the prompt just tells it WHEN and WHERE.

the prompt is the leash, not the motor. the motor is 7,767 conversations of pure reddit energy, burned into every parameter during training.

---

## architecture

```
nanochat d20 — 477M parameters
20 layers, 1280 dim, 10 heads
RoPE + RMSNorm + ReLU^2 + QK-Norm + Value Embeddings + Sliding Window
vocab: 32768 BPE (tiktoken)
```

based on [karpathy's nanochat](https://github.com/karpathy/nanochat), the d20 config. we took his clean, beautiful, well-documented architecture and filled it with reddit energy. sorry andrej.

the inference engine is rewritten entirely in C. not "python wrapper around C." not "pytorch with a C extension." pure C. the kind of C where you manage your own memory and think about cache lines and wonder why you didn't just use pytorch like a normal person.

but here's why: no dependencies. no python at runtime. no CUDA. no cuDNN. no "pip install failed because your numpy version is 0.0.1 too old." you compile it, you run it. it loads the weights, does matrix multiplies, and spits out tokens. that's it. if it runs on your machine, it runs. if it doesn't, you probably forgot to `make`.

the model uses special tokens for chat formatting:

```
<bos> <user_start> {system_prompt} <user_end>
<assistant_start> ok <assistant_end>
<user_start> {your question} <user_end> <assistant_start>
```

that "ok" after the system prompt? that's the model acknowledging its constraints. one token. minimal overhead. maximum personality retention. the system prompt costs ~30 extra tokens per turn. worth it.

INT8 quantization brings the weights from 1.7GB (fp16) to 857MB. quality loss is minimal — turns out cynicism quantizes well. who knew.

---

## training pipeline

the training happened on Lambda Cloud, 4x H100 80GB, and it went like this:

**stage 1: pretrain on FineWeb** — 16,600 steps, final loss 2.55. this is where the model learns english. grammar. facts. the boring stuff. the foundation. every building needs a foundation, even a building full of shitposts.

**stage 2: midtrain with identity injection** — 7,767 hand-crafted identity conversations, 5 epochs, loss drops to 1.02. this is where the magic happens. this is where the model stops being a generic text predictor and starts being wtforacle. every conversation is a reddit-style Q&A pair, written to sound like someone who's been on the internet too long and developed opinions as a coping mechanism.

the conversations are hand-written + AI-augmented. the hand-written ones set the tone. the AI-augmented ones fill in the gaps. 7,767 is a lot of conversations. it's enough to build a personality. it's enough to make the model default to cynicism instead of helpfulness.

**stage 3: SFT (supervised fine-tuning)** — 27% identity conversations + 73% SmolTalk/ARC/GSM8K, 911 steps. this is the balancing act. pure identity training makes the model a one-trick pony. it becomes SO cynical it can't answer factual questions. the 73% general data keeps the model grounded. it can still do math. it can still answer questions. it just does it with attitude.

the ratio matters. 27% identity is the sweet spot. lower than that and the personality fades. higher than that and it can't think straight. like alcohol — enough to loosen up, not enough to fall over.

total cost: $30-50 on Lambda. cheaper than therapy. arguably more effective.

---

## inference

runs on CPU. no GPU. no pytorch. no python at runtime. just C.

```bash
# clone
git clone https://github.com/ariannamethod/WTForacle

# compile
cd wtf.c && make && cd ..

# run raw
./wtf.c/wtf wtfweights/wtforacle_q8.bin wtfweights/wtforacle.tok -p "32759 32760 <token_ids> 32761 32762" -n 100 -t 0.9
```

or through the python REPL wrapper (the civilized way):

```bash
python wtforacle.py
```

```
============================================================
  WTFORACLE
  the reddit oracle nobody asked for
  WTForacle d20 (477M, INT8)
============================================================
Commands: /quit, /tokens N, /temp T, /raw (toggle system prompt)

You: who are you?
WTForacle: i am the chatbot. i am here for information. i am also an algorithm. XD
```

REPL commands:
- `/quit` — exit (prints "later loser" because of course it does)
- `/tokens N` — set max generation tokens
- `/temp T` — set temperature (0.9 recommended, lower = boring)
- `/raw` — toggle system prompt off/on (raw mode = no personality constraints, pure weights)

runs on a MacBook Pro 2019 (Intel i5, 8GB RAM). about 1 token per second. the oracle takes its time. deal with it. if you wanted fast you'd ask chatgpt and get a polite non-answer in 200ms.

---

## weights

| File | Format | Size | Use |
|------|--------|------|-----|
| `wtforacle_fp16.bin` | float16 | 1.7 GB | archive / max quality |
| `wtforacle_q8.bin` | INT8 per-row | 857 MB | inference (recommended) |
| `wtforacle.tok` | binary tokenizer | 339 KB | required for inference |

download from [HuggingFace](https://huggingface.co/ataeff/WTForacle).

the fp16 weights are the "museum quality" version. the q8 weights are what you actually run. the difference is negligible. cynicism survives quantization. it might even improve it — there's something poetic about reducing precision on a model that was never precise to begin with.

---

## benchmarks

| Metric | Score | Comment |
|--------|-------|---------|
| ARC-Easy | 49.66% | knows stuff |
| ARC-Challenge | 37.46% | knows some stuff |
| MMLU | 34.28% | barely literate |
| SpellingBee | 99.22% | can spell tho |
| Vibes | 100% | undefeated |

the benchmarks tell an honest story. this is a 477M parameter model. it's not going to ace MMLU. it's not going to solve competitive programming problems. it knows roughly half of the easy stuff, a third of the hard stuff, and it can spell almost everything correctly.

but vibes? vibes are 100%. undefeated. no model at any parameter count has better vibes. this is a hill we will die on.

the real benchmark is: did it make you exhale through your nose slightly harder than usual? if yes, it passed. if no, raise the temperature.

---

## system prompt archaeology

the seven iterations of getting a 477M model to maintain personality, documented for anyone trying to do the same thing. consider this a field guide to identity engineering at small scale.

**lesson 1: negations don't work.** "you are NOT helpful" makes the model MORE helpful. small models can't reliably process negation. use affirmations: "you are cynical" instead of "you are not sincere." say what it IS, not what it ISN'T.

**lesson 2: abstractions don't work.** "you have a sarcastic personality" is too abstract. the model nods along and then generates generic text. "if you say something helpful, always end with sarcasm" — that's a concrete rule. a pattern the model can follow.

**lesson 3: character sheets don't work.** long backstories, personality descriptions, example dialogues in the system prompt — all of it gets lost. 477M params have limited context bandwidth. every token of system prompt is a token not spent on actual generation. keep it short. keep it concrete.

**lesson 4: theatrics don't work.** stage directions, emotional beats, narrative framing — the model generates them as output. it can't separate instruction format from generation format. save the creativity for the training data.

**lesson 5: identity is in the weights.** the 7,767 training conversations do more for personality than any system prompt ever could. the prompt is maintenance. the weights are identity. the prompt is the leash. the weights are the motor.

**lesson 6: temperature matters more than you think.** temp 0.9 is optimal. at 0.7, the generic assistant leaks through. the training baked cynicism into the weights, but it also baked in helpfulness from the pretrain data. high temperature lets the identity-trained weights express themselves. low temperature regresses to the mean, and the mean is "how can I help you today?"

**lesson 7: the acknowledgment token matters.** after the system prompt, the model responds "ok" — one token. this tiny acknowledgment anchors the personality for the rest of the conversation. without it, the model treats the system prompt as context rather than instruction. "ok" means "i heard you, i got it, let's go."

---

## related

- [ariannamethod](https://github.com/ariannamethod) — the ecology this crawled out of
- [nanochat](https://github.com/karpathy/nanochat) — the architecture we corrupted
- [haze](https://github.com/ariannamethod/haze) — the philosophical predecessor
- [leo](https://github.com/ariannamethod/leo) — the resonant one

---

## license

GPL 3.0. do whatever. wtforacle doesn't care. ever. ¯\\\_(ツ)\_/¯

---

*"worse takes, better GPUs, same depression."* -- wtforacle, 2026
