# Mini-LLM
my own chat-gpt? Yes, but a small one!!
# Mini LLM From Scratch (Pure PyTorch)

A minimal language model built from scratch using **raw PyTorch tensors** to understand how modern LLMs actually work under the hood.

No `nn.Module`, no high-level abstractions — just the core ideas implemented step by step.

---

## What This Project Does

- Trains a small language model on **Alice in Wonderland**
- Learns to **predict the next word** from previous words
- Generates new text one word at a time

This project is meant for **learning**, not for performance or production.

---

## Why This Project Exists

Most tutorials hide the logic behind layers and libraries.

This project answers questions like:
- How does a language model turn words into numbers?
- How does attention decide *which* words matter?
- How is the next word chosen?
- Where does learning actually happen?

Everything is written explicitly so the flow is easy to follow.

---

## Model Overview

The model follows the same basic structure used in real LLMs:


Each part is implemented manually to make the logic clear.

---

## Key Components

### Token & Position Embeddings
- Converts words into vectors
- Adds information about word order

### Self-Attention
- Allows each word to look at previous words
- Helps the model focus on relevant context
- Uses a causal mask so the model cannot see the future

### Feed Forward Network
- Adds non-linearity
- Helps the model learn complex patterns

### Output Layer
- Converts internal vectors into scores for every word
- The highest probability word is selected as the next token

---

## Training Process

- Random windows of text are sampled
- The model predicts the next word at each position
- Error is measured using **cross-entropy loss**
- Parameters are updated using **manual gradient descent**

This loop repeats until the model improves.

---

## Text Generation

After training:
1. A few starting words are given
2. The model predicts the next word
3. That word is added back to the input
4. The process repeats to generate text

This is exactly how real LLMs generate text — just at a much smaller scale.

