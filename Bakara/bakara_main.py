import time

import cv2
import numpy as np
import scipy.signal

PATH = "C:\\Users\\t8670535\\Dropbox\\Talpiot\\SemesterC\\Projecton\\Puzzle-Solver\\image_processing\\frame{0}.png"
N = 88
def main():
    images = [cv2.imread(PATH.format(i)) for i in range(N)]
    for im in images:
        vect = get_motion_vect(im)

def debug_get_motion_vect(im, thresh = 15):
    # SOBEL GRAY IMAGE
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)[:, :, 0]
    sobelx_pos = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=31)
    sobelx_neg = sobelx_pos.copy()

    # POS
    sobelx_pos[sobelx_pos < 0] = 0
    sobelx_pos = np.abs(sobelx_pos) / np.max(sobelx_pos)
    retvalp, thrashed_pos = cv2.threshold(sobelx_pos, 0.15, 1, cv2.THRESH_TOZERO)

    # NEG
    sobelx_neg[sobelx_neg > 0] = 0
    sobelx_neg = np.abs(sobelx_neg) / np.max(np.abs(sobelx_neg))
    retvaln, thrashed_neg = cv2.threshold(sobelx_neg, 0.15, 1, cv2.THRESH_TOZERO)

    # SHOW IMAGES OF SOBELS
    cv2.imshow("pos", cv2.resize(thrashed_pos, dsize=None, fx=0.2, fy=0.2))
    cv2.imshow("neg", cv2.resize(thrashed_neg, dsize=None, fx=0.2, fy=0.2))
    cv2.waitKey(0)

    # CROSS CORRELATE
    t1 = time.perf_counter()
    cc = scipy.signal.convolve(thrashed_pos, np.flip(thrashed_neg, axis=(0, 1)), mode='same')
    print(time.perf_counter() - t1)
    # auto[int(0.4*auto.shape[0]):int(0.6*auto.shape[0]), int(0.4*auto.shape[1]):int(0.6*auto.shape[1])] = 0
    # cc[int(0.4*cc.shape[0]):int(0.6*cc.shape[0]), :] = 0
    cc = cc/np.max(cc)

    cv2.imshow("auto", cv2.resize(cc, dsize=None, fx=0.2, fy=0.2))
    cv2.waitKey(0)

    # ARGMAX AND VECTORS
    offset = np.flip(np.unravel_index(np.argmax(cc[:cc.shape[0]//2, :]), cc.shape))
    center = np.flip([cc.shape[0] // 2, cc.shape[1] // 2])
    vect = center - offset

    ak1 = center + vect
    ak2 = center - vect

    # PRINTS
    print(cc.shape)
    retval, thrashed_cc = cv2.threshold(cc, 0.95, 1, cv2.THRESH_TOZERO)
    thrashed_cc[int(0.49*cc.shape[0]):int(0.51*cc.shape[0]), int(0.49*cc.shape[1]):int(0.51*cc.shape[1])] = 1
    cv2.circle(thrashed_cc, tuple(ak1), 40, 1, 5)
    cv2.circle(thrashed_cc, tuple(ak2), 40, 1, 5)
    cv2.circle(thrashed_cc, tuple(center), 40, 1, 5)

    cv2.arrowedLine(thrashed_cc, tuple(center), tuple(ak1), 1, 5)

    cv2.imshow("correlation with derv", cv2.resize(thrashed_cc*3/4 + thrashed_pos/4 + thrashed_neg/4, dsize=None, fx=0.2, fy=0.2))

    # Do not attempt to touch
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_motion_vect(im, thresh = 0.15):
    # SOBEL GRAY LAB CIE IMAGE
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)[:, :, 0]
    sobelx_pos = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=31)
    sobelx_neg = sobelx_pos.copy()

    # POS SOBEL
    sobelx_pos[sobelx_pos < 0] = 0
    sobelx_pos = np.abs(sobelx_pos) / np.max(sobelx_pos)
    retvalp, thrashed_pos = cv2.threshold(sobelx_pos, thresh, 1, cv2.THRESH_TOZERO)

    # NEG SOBEL
    sobelx_neg[sobelx_neg > 0] = 0
    sobelx_neg = np.abs(sobelx_neg) / np.max(np.abs(sobelx_neg))
    retvaln, thrashed_neg = cv2.threshold(sobelx_neg, thresh, 1, cv2.THRESH_TOZERO)

    # CROSS CORRELATE
    cc = scipy.signal.convolve(thrashed_pos, np.flip(thrashed_neg, axis=(0, 1)), mode='same')
    cc = cc / np.max(cc)

    # ARGMAX AND VECTORS
    offset = np.flip(np.unravel_index(np.argmax(cc[:cc.shape[0] // 2, :]), cc.shape))
    center = np.flip([cc.shape[0] // 2, cc.shape[1] // 2])
    vect = center - offset

    # RET VAL
    return vect

if __name__ == "__main__":
    main()