import cv2
import numpy as np


def rgb2gray(obs):
    return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[None]

def normalize(obs):
    return obs / 255.0

def hwc2chw(obs):
    # return obs.permute(0, 3, 1, 2)
    return np.transpose(obs, [2, 0, 1])

def tofloat32(obs):
    return obs.astype(np.float32)

def preprocess(obs):
    # return normalize(hwc2chw(rgb2gray(obs)))
    return tofloat32(normalize(rgb2gray(obs)))