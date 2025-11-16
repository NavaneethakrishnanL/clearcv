import numpy as np
from clearcv.utils.kernels import gaussian_kernel, convolve2d


def resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Resize using simple nearest-neighbor interpolation."""
    h, w = img.shape[:2]

    row_scale = h / new_h
    col_scale = w / new_w

    out = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            y = int(i * row_scale)
            x = int(j * col_scale)
            out[i, j] = img[y, x]
    return out


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around center using nearest neighbor."""
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):

            x = (j - cx) * cos_a + (i - cy) * sin_a + cx
            y = -(j - cx) * sin_a + (i - cy) * cos_a + cy

            if 0 <= x < w and 0 <= y < h:
                out[i, j] = img[int(y), int(x)]

    return out


def blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur using convolution."""
    if img.ndim == 2:
        kernel = gaussian_kernel(ksize, sigma)
        return convolve2d(img, kernel)

    out = np.zeros_like(img)
    kernel = gaussian_kernel(ksize, sigma)
    for c in range(3):
        out[..., c] = convolve2d(img[..., c], kernel)

    return out


def crop(img: np.ndarray, top: int, left: int, bottom: int, right: int) -> np.ndarray:
    """Crop an image: (y1:y2, x1:x2)."""
    return img[top:bottom, left:right]


def pad(img: np.ndarray, pad_h: int, pad_w: int, mode="constant", value=0) -> np.ndarray:
    """Pad an image on all sides."""
    if mode == "constant":
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant", constant_values=value)
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=mode)


def flip(img: np.ndarray, direction="horizontal") -> np.ndarray:
    """Flip horizontally or vertically."""
    if direction == "horizontal":
        return img[:, ::-1]
    elif direction == "vertical":
        return img[::-1]
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
