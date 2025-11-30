# Vision Transformer (ViT) From Scratch

This repository contains a basic implementation of the Vision Transformer (ViT) model built from scratch using PyTorch. The goal of this project was to understand and implement the main components of ViT without using high-level transformer libraries. The model was trained and tested on the CIFAR-10 dataset.

## What’s Included

- Patch embedding (splitting images into patches and flattening them)
- Positional encoding
- Multi-head self-attention (implemented manually)
- Transformer encoder blocks (LayerNorm + MLP)
- Classification token (CLS)
- Training and evaluation pipeline for CIFAR-10

## Why This Project

The Vision Transformer shows that transformers can work well on images if the images are broken into patches. This project was done to learn how ViT works internally by building everything step by step.

## Training Details

- Dataset: CIFAR-10  
- Optimizer: AdamW  
- Loss: CrossEntropy  
- Epochs: configurable  
- Works on both CPU and GPU  

The model reaches accuracy comparable to standard CNN models when trained properly.



