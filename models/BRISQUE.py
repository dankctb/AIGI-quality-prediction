import torch
from torch import nn
import torch.nn.functional as F
from .utils import compute_image_mscn_transform, extract_subband_feats

import numpy as np
import cv2
import glob
import pandas as pd

# Matlab Interpolation Implementation in Python
from .imresize import imresize


def brisque(image):
    y_mscn = compute_image_mscn_transform(image)
    half_scale = imresize(image, scalar_scale=0.5, method='bicubic')
    y_half_mscn = compute_image_mscn_transform(half_scale)
    feats_full = extract_subband_feats(y_mscn)
    feats_half = extract_subband_feats(y_half_mscn)
    return np.concatenate((feats_full, feats_half))


class BRISQUE(nn.Module):
    def __init__(self):
        super(BRISQUE, self).__init__()

    def forward(self, x):

        return brisque(x)