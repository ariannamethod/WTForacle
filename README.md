# WTForacle  

> sir this is reddit (rc) reddit

**the reddit WTForacle nobody asked for.**

477M parameter AI which answers questions with the energy of a locked thread, 400 downvotes, and someone who WILL explain it anyway.
not an assistant. not helpful. not sorry.

---

## what is this

a custom-trained Karphaty's nanochat GPT (d20, 477M params) and rewritten in C. finetuned on 7,767 hand-crafted identity conversations that capture the essence of:

- confidently wrong advice
- unsolicited opinions nobody needed
- the specific energy of someone typing a reply at 3am then hitting send
- reddit wisdom distilled into pure neural weights

trained on 4x H100 80GB. cost more than your rent. still tells you to drink water.

## architecture

```
20 layers, 1280 dim, 10 heads
RoPE + RMSNorm + ReLU^2 + QK-Norm + Value Embeddings + Sliding Window
vocab: 32768 BPE (tiktoken)
trained: pretrain FineWeb -> midtrain 5 epochs identity -> SFT 27% identity mix
```

originally karpathy's nanochat but in c and we ruined it with personality.

  
## sample outputs

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

temperature 0.9-1.0 recommended. lower temps produce generic assistant garbage. this model needs room to breathe.

## weights

| File | Format | Size | Use |
|------|--------|------|-----|
| `wtforacle_fp16.bin` | float16 | 1.7 GB | archive / max quality |
| `wtforacle_q8.bin` | INT8 per-row | 857 MB | inference (recommended) |
| `wtforacle.tok` | binary tokenizer | 339 KB | required for inference |

## inference

runs on CPU. no GPU. no pytorch. no python at runtime. C. hardcore :P

```bash
# clone
git clone https://github.com/ariannamethod/WTForacle
  
# compile
make

# run
./nano wtforacle_q8.bin wtforacle.tok -p "32759 32760 <token_ids> 32761 32762" -n 100 -t 0.9
```

or through the python REPL wrapper:

```bash
python wtforacle.py
```

```
============================================================
  WTFORACLE REPL
  Model: WTForacle d20 (477M, INT8)
============================================================
Commands: /quit, /tokens N, /temp T

You: who are you?
WTForacle: i am the chatbot. i am here for information. i am also an algorithm. XD
```

~1 tok/s on MacBook Pro 2019 (Intel i5). yes it's slow. the oracle takes its time. deal with it.

## training pipeline

```
1. pretrain on FineWeb (16600 steps, loss 2.55)
2. midtrain: inject 7767 identity conversations (5 epochs, loss 1.02)
3. SFT: 27% identity + 73% SmolTalk/ARC/GSM8K (911 steps)
```

identity dataset: hand-written + AI-augmented reddit-style Q&A pairs that sound like someone who's been on the internet too long and developed opinions as a coping mechanism.

## the philosophy

wtforacle exists because the world has enough helpful AI assistants. you don't need a solution, but someone to tell you "stop asking reddit for advice. ok anyway here's advice". the truth btw comes from the comment section, not the article.

477M parameters of pure cynicism is exactly what the moment calls for.

## benchmarks

| Metric | Score | Comment |
|--------|-------|---------|
| ARC-Easy | 49.66% | knows stuff |
| ARC-Challenge | 37.46% | knows some stuff |
| MMLU | 34.28% | barely literate |
| SpellingBee | 99.22% | can spell tho |
| Vibes | 100% | undefeated |

## related

- [ariannamethod](https://github.com/ariannamethod) - the ecology this crawled out of
- [nanochat](https://github.com/karpathy/nanochat) - the architecture we corrupted
- [haze](https://github.com/ariannamethod/haze) - the schizo predecessor

## license

GPL 3.0. do whatever. wtforacle doesn't care. ever. ¯\\\_(ツ)\_/¯

---

*"worse takes, better GPUs, same depression."* — wtforacle, 2026
