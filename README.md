# Mistral 7B - LoRA Merged Fine-Tuned Model

This repository hosts a **merged LoRA fine-tuned version** of [Mistral-7B-v0.1](https://huggingface.co/Hirudika/mistral7b-lora-merged), trained by `Hirudika` for domain-specific tasks such as health insurance Q&A or similar applications. The LoRA weights are **merged** with the base model — no separate adapter is needed.

---

## Model Card

- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Technique**: LoRA (Low-Rank Adaptation)
- **Merged**: (can be used as a standalone model)
- **Intended Use**: Domain-specific chat, Q&A, instruction-following

---

## Installation

First, ensure your environment is set up:

```bash
pip install -U transformers accelerate bitsandbytes
