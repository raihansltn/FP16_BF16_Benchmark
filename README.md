# FP16 vs BF16 Precision Study in GPT-2

Inspired by paper: [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/pdf/2510.26788)
Repo: [Precision-RL](https://github.com/sail-sg/Precision-RL)

This project reproduces and visualizes the effects of reduced-precision arithmetic (FP16 and BF16) on GPT-2 inference. It follows insights from the paper [Defeating the Training-Inference Mismatch via FP16](http://arxiv.org/pdf/2510.26788) to empirically verify numerical stability across different floating-point formats.

## Overview
The notebook:
- Samples text prompts from the [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) dataset.
- Runs inference with GPT-2 under FP32, FP16, and simulated BF16.
- Computes **Mean Squared Error (MSE)** and **Pearson correlation** across precision modes.
- Visualizes distribution alignment between Hugging Face and simulated vLLM probabilities.

## Key Finding
FP16 and BF16 produce nearly identical token probability distributions (MSE < 1e-8, R ≈ 1.0), confirming that FP16 can accelerate inference **without measurable accuracy loss**.

![Output.](/assets/output.png)
![Metrics.](/assets/metrics.jpg)

## Requirements
```bash
pip install torch transformers datasets numpy matplotlib scipy
```