import numpy as np
from clearcv.metrics import ssim

def test_ssim_identical_images():
    img = np.ones((100, 100)) * 127
    score = ssim(img, img)
    assert score == 1.0

def test_ssim_small_difference():
    img1 = np.zeros((50, 50))
    img2 = np.zeros((50, 50))
    img2[10:20, 10:20] = 20  # small patch
    score = ssim(img1, img2)
    assert 0 <= score < 1.0

def test_ssim_value_range():
    img1 = np.random.randint(0, 255, (64, 64)).astype(np.uint8)
    img2 = np.random.randint(0, 255, (64, 64)).astype(np.uint8)
    score = ssim(img1, img2)
    assert -1 <= score <= 1
