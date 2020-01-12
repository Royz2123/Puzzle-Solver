import cv2
import numpy as np
from constants import *

import math


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def dist_from_line(next_corner, point):
    x0, y0 = point[0], point[1]
    x2, y2 = next_corner[0], next_corner[1]

    denom = np.sqrt(x2**2 + y2**2)
    neumon = y2*x0 - x2*y0

    return neumon / denom



def relevant_section(img):
    return img[-900:-200,-650:-200]


def output(name, img):
    cv2.imshow(name, img)
    cv2.imwrite(RESULTS_BASE + name + ".png", img)


def get_test_images():
    above = cv2.imread(ABOVE, 1)
    below = cv2.imread(BELOW, 0)

    above = relevant_section(above)
    below = relevant_section(below)

    return above, below
