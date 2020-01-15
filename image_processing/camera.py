import cv2
import numpy as np
import matplotlib.pyplot as plt


DIM=(3264, 2448)
K=np.array([[2107.8678504205704, 0.0, 1544.8044927219369], [0.0, 2118.630372303031, 1234.9817473763783], [0.0, 0.0, 1.0]])
D=np.array([[-0.12271606120274124], [0.2074119120947846], [-0.4160633486018614], [0.2661144859244342]])

# Create a VideoCapture object and read from input file
try:
    STILL_CAP = cv2.VideoCapture(0)
    STILL_CAP.set(cv2.CAP_PROP_FRAME_WIDTH, DIM[0])
    STILL_CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, DIM[1])

    # Check if camera opened successfully
    if (STILL_CAP.isOpened() == False):
        print("Still Camera Disconnected!")
except:
    print("Still Camera Disconnected!")


def undistort(img):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def take_picture(still_camera=True):
    ret, frame = STILL_CAP.read()
    if ret:
        return undistort(frame)
    else:
        return None

