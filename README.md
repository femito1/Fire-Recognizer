# Fire Detection with Fine-Tuned ResNet18

This project demonstrates how to fine-tune a [ResNet18](https://arxiv.org/abs/1512.03385) convolutional neural network to recognize fires in images. By leveraging transfer learning and a custom fire/non-fire image dataset, we can achieve robust performance with relatively little training data and time.

## Overview

- **Goal:** Classify images into "fire" vs. "no-fire".
- **Model Architecture:** Pre-trained ResNet18 fine-tuned on a custom dataset.
- **Framework:** [PyTorch](https://pytorch.org/)
- **Key Techniques:** Transfer learning, data augmentation, and custom training scripts.

## Features

- **Pretrained Weights:** Utilizes ImageNet-pretrained ResNet18 weights for faster convergence and higher accuracy.
- **Flexible Training:** Easily adjust hyperparameters (learning rate, batch size, epochs) in the provided script.
- **Evaluation Metrics:** Automatically computes accuracy, precision, recall, and F1-score on the validation/test set.
- **Reproducibility:** Code is structured for reproducibility and easy adaptation to new datasets.

## Setup & Requirements

**Environment:**
- Python 3.8+
- PyTorch (>=1.7.0) and TorchVision
- CUDA-capable GPU (recommended but not required)
- Other packages: `numpy`, `matplotlib`, `tqdm`
