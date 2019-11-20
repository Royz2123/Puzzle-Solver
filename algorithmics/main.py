import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in grayscale
img = cv2.imread('./images/pieces.jpeg')

cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()