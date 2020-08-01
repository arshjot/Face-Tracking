import numpy as np
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin
import numpy as np


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def resize_image(im, size):
    """
    Resizes image to the given size while maintaining aspect ratio.
    Size argument corresponds to the output size of the smallest dimension.
    """

    h, w = im.shape[:2]
    small_dim = np.argmin([h, w])
    small_dim_out_s = int(size)
    large_dim_out_s = int(small_dim_out_s * (max(h, w) / min(h, w)))
    out_s = (small_dim_out_s, large_dim_out_s) if small_dim == 0 else (large_dim_out_s, small_dim_out_s)

    return cv2.resize(im, out_s[::-1])
