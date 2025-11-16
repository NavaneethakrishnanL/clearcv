import numpy as np
from clearcv.metrics.psnr import psnr


def test_psnr_identical_images():
    img = np.ones((10, 10, 3), dtype=np.uint8) * 128
    assert psnr(img, img) == float("inf")


def test_psnr_known_value():
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 10
    
    value = psnr(img1, img2)
    assert value > 20  # general sanity check
