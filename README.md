# Mistral 7B - LoRA Merged Fine-Tuned Model

This repository contains the **merged** fine-tuned version of the [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) model using **LoRA**. It is fine-tuned for health insurance Q&A tasks (or your specific domain – edit as necessary) and ready for inference without requiring additional adapters.

## Installation

Make sure you have Python 3.8+ and the latest Hugging Face libraries.

### 1. Install dependencies

```bash
pip install -U transformers accelerate peft bitsandbytes
