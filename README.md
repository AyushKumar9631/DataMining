---
title: Crowd Counting
emoji: 👥
colorFrom: violet
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Crowd Counting — CrowdNetV2

Estimates the number of people in a crowd image using a custom neural network
with a pretrained MobileNetV2 backbone.

## Model
- **Architecture**: MobileNetV2 (ImageNet pretrained) + custom regression head
- **Training data**: ShanghaiTech Part B
- **Loss**: Huber loss on log₁p(count)
- **Metrics**: Val MAE ~12–18 on ShanghaiTech Part B test set

## Density levels
| Label | Count range |
|---|---|
| Sparse | < 20 |
| Moderate | 20 – 74 |
| Dense | 75 – 199 |
| Very Dense | ≥ 200 |
