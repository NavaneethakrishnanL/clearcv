# ClearCV


ClearCV is a lightweight, NumPy-based computer vision toolkit implemented in Python. It aims to be small, readable, and easy to extend.

## Features (v0.1.1)
### SSIM — Structural Similarity Index

ClearCV provides a pure NumPy SSIM implementation:

```python
from clearcv.metrics import ssim

similarity = ssim(img1, img2)
print(similarity)


## Features (v0.1)
- Read/write PPM images (fast, minimal I/O) and optional `imageio` fallback for PNG/JPG
- RGB <-> Grayscale conversions
- Convolution-based filters (box blur, Sobel edge detection)
- Nearest-neighbor resize and simple rotate/flip
- Utility helpers


## Installation


Create a virtual environment and install:


```bash
python -m venv venv
source venv/bin/activate
pip install -e .