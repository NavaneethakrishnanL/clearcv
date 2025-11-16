import numpy as np
from clearcv.ops import (
    resize, rotate, flip, crop, pad, blur, psnr, ssim
)


def test_resize():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = resize(img, (20, 20))
    assert out.shape == (20, 20)


def test_rotate():
    img = np.zeros((10, 20), dtype=np.uint8)
    out = rotate(img, 90)
    assert out.shape == (20, 10)


def test_flip():
    img = np.arange(9).reshape(3, 3)
    out_h = flip(img, mode="horizontal")
    out_v = flip(img, mode="vertical")

    assert np.array_equal(out_h, img[:, ::-1])
    assert np.array_equal(out_v, img[::-1])


def test_crop():
    img = np.arange(100).reshape(10, 10)
    out = crop(img, 2, 2, 5, 5)
    assert out.shape == (5, 5)
    assert out[0, 0] == img[2, 2]


def test_pad():
    img = np.zeros((5, 5))
    out = pad(img, pad_width=2, value=1)
    assert out.shape == (9, 9)
    assert (out[:2, :] == 1).all()
    assert (out[-2:, :] == 1).all()


def test_blur():
    img = np.zeros((20, 20))
    img[10, 10] = 255  # single bright pixel
    out = blur(img, ksize=5)
    assert out.shape == img.shape
    assert out[10, 10] < 255  # blur effect


def test_psnr():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.zeros((10, 10), dtype=np.uint8)

    assert psnr(img1, img2) == float('inf')

    img2[0, 0] = 255
    value = psnr(img1, img2)
    assert value < 20  # high error → low PSNR


def test_ssim():
    img1 = np.ones((20, 20), dtype=np.uint8) * 100
    img2 = np.ones((20, 20), dtype=np.uint8) * 100

    assert ssim(img1, img2) == 1.0  # identical images

    img2[10, 10] = 0
    value = ssim(img1, img2)
    assert 0 <= value <= 1
