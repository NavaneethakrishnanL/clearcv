import numpy as np


def gaussian_kernel(size: int, sigma: float):
    """Generate a Gaussian blur kernel."""
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def box_kernel(size: int):
    """Generate a normalized box (mean) filter."""
    return np.ones((size, size), dtype=float) / (size * size)


def sharpen_kernel():
    return np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ], dtype=float)


def edge_kernel():
    return np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)


def sobel_x():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=float)


def sobel_y():
    return np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ], dtype=float)

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform a 2D convolution manually (dependency-free).

    Args:
        image: 2D or 3D NumPy array (H×W or H×W×C)
        kernel: 2D convolution kernel

    Returns:
        Convolved image as float32 NumPy array.
    """

    # Handle multi-channel images (RGB)
    if image.ndim == 3:
        return np.stack([convolve2d(image[..., c], kernel) for c in range(image.shape[-1])], axis=-1)

    # Flip kernel (true convolution)
    kernel = np.flipud(np.fliplr(kernel))

    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad image with zeros
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    output = np.zeros((h, w), dtype=np.float32)

    # Convolution loop
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output

