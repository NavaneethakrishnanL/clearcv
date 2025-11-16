import numpy as np
from scipy.ndimage import gaussian_filter


def ssim(img1: np.ndarray, img2: np.ndarray, K1=0.01, K2=0.03, win_size=11, sigma=1.5, L=255):
    """
    Compute Structural Similarity Index (SSIM) between two grayscale images.

    Args:
        img1 (np.ndarray): First grayscale image (H × W)
        img2 (np.ndarray): Second grayscale image (H × W)
        K1, K2: Stability constants
        win_size: Gaussian window size
        sigma: Gaussian kernel sigma
        L: Dynamic range (255 for uint8)

    Returns:
        float: SSIM score (−1 to 1)
    """

    # convert to float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Gaussian kernel filtering
    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2

    # stability constants
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return np.mean(ssim_map)
