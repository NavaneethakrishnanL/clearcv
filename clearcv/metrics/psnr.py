import numpy as np


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR)."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float("inf")

    return 20 * np.log10(max_val) - 10 * np.log10(mse)
