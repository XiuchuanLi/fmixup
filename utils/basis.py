from .dct import dct_2d, idct_2d
import torch
import numpy as np
from PIL import Image


def dct_8_8(img, mask):
    fimg = torch.zeros_like(img)
    for i in range(0, img.shape[-2], 8):
        for j in range(0, img.shape[-1], 8):
            fimg[:, i:(i+8), j:(j+8)] = dct_2d(img[:, i:(i+8), j:(j+8)]) * mask
    return fimg


def idct_8_8(fimg):
    iimg = torch.zeros_like(fimg)
    for i in range(0, fimg.shape[-2], 8):
        for j in range(0, fimg.shape[-1], 8):
            iimg[:, i:(i+8), j:(j+8)] = idct_2d(fimg[:, i:(i+8), j:(j+8)])
    return iimg


def get_mask(frequency_range):
    upper = int(64 * frequency_range)
    mask = torch.zeros([8, 8])
    s = 0
    while upper > 0:
        for i in range(min(s+1, 8)):
            for j in range(min(s+1, 8)):
                if i + j == s:
                    if s % 2 == 0:
                        mask[j, i] = 1
                    else:
                        mask[i, j] = 1
                    upper -= 1
                    if upper == 0:
                        return mask
        s += 1
    return mask


def norm(x):
    return (torch.sum(x ** 2) ** 0.5).item()


def schmidt(v, ortho_with):
    v_repeat = torch.stack([v, ] * len(ortho_with), dim=0)
    coefficient = (ortho_with * v_repeat).flatten(1).sum(1)
    project = coefficient.reshape([-1, 1, 1, 1]) * ortho_with
    v = v - torch.sum(project, dim=0)
    return v / norm(v)

