# Echo_124M
# 🧠 Echo-124M: A Minimal GPT-2 Style Language Model

**Echo-124M** is a lightweight implementation of a **GPT-2 124M parameter model**, built and trained using the **NanoGPT framework**.  
It replicates the **architecture and behavior of GPT-2 (small)** while remaining **minimal, efficient, and easy to train on limited hardware**.

---

## 🚀 Overview

This project implements a **transformer-based autoregressive language model** similar to **OpenAI’s GPT-2 (124M)** using **NanoGPT**.  
It includes:
- Transformer blocks (multi-head self-attention + feedforward layers)  
- Layer normalization and residual connections  
- Token and positional embeddings  
- Causal attention mask for autoregressive prediction  

You can use it for:
- Text generation  
- Fine-tuning on custom datasets  
- Understanding transformer internals  

---

## 🧩 Model Architecture

| Component | Description |
|------------|-------------|
| **Model Type** | GPT-2 (124M equivalent) |
| **Layers** | 12 Transformer blocks |
| **Embedding Size** | 768 |
| **Attention Heads** | 12 |
| **Total Parameters** | ~124 Million |
| **Context Window** | 1024 tokens |
| **Activation** | GELU |
| **Optimizer** | AdamW |
| **Training Framework** | NanoGPT (PyTorch) |

---

## 📦 Project Structure

