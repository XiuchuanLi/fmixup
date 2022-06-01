import torch
import math
import numpy as np
from matplotlib import pyplot as plt


basis = torch.zeros([56, 56, 224, 224])
for u in range(56):
    for v in range(56):
        img = np.zeros([57, 57])
        img[u, v] = 1
        for i in range(max(0, 4*u-3), 4*u+4):
            for j in range(max(0, 4*v-3), 4*v+4):
                float_scrx = i * (56/224)
                float_srcy = j * (56/224)
                int_scrx = math.floor(float_scrx)
                int_scry = math.floor(float_srcy)
                dec_scrx = float_scrx - int_scrx
                dec_scry = float_srcy - int_scry
                basis[u, v, i, j] = (1-dec_scrx)*(1-dec_scry)*img[int_scrx, int_scry] + \
                                    dec_scrx*(1-dec_scry)*img[int_scrx+1, int_scry] + \
                                    (1 - dec_scrx)*dec_scry*img[int_scrx, int_scry+1] + \
                                    dec_scrx*dec_scry*img[int_scrx+1, int_scry+1]


def bilinear(K):
    scr_noise = torch.randn([K, 3, 56, 56])
    dst_noise = torch.zeros([K, 3, 224, 224])
    for i in range(K):
        for j in range(3):
            temp_scr_noise = scr_noise[i, j].reshape([56, 56, 1, 1])
            dst_noise[i, j] = torch.sum(temp_scr_noise * basis, dim=[0, 1])
    return dst_noise

