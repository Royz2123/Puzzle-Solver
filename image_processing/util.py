import cv2
import numpy as np
import math

from image_processing.constants import *


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


def relevant_section(img, test=True):
    # for 2 pieces
    # tmp = img[-900:-200,-650:-200]

    # for 34 pieces
    # tmp = img[70:-100,300:-230]

    if test:
        # for 4 pieces
        tmp = img[100:-100, 200:-230]
    else:
        # for prosak pieces
        tmp = img[260:-395, 810:-1100]

    cv2.imshow("tmp",cv2.resize(tmp, None, fx=0.2, fy=0.2))
    # cv2.waitKey(0)
    return tmp


def output(name, img):
    cv2.imshow(name, cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(RESULTS_BASE + name + ".png", img)


def get_test_images():
    above = cv2.imread(ABOVE, 1)
    below = cv2.imread(BELOW, 0)

    return above, below
