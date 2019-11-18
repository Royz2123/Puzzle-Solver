import numpy as np
import cv2

BASE = "./test_images/"
ABOVE = BASE + "above.jpeg"
BELOW = BASE + "below.jpeg"

above = cv2.imread(ABOVE,0)
below = cv2.imread(BELOW,0)

cv2.imshow('image', below)
k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()

#
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',img)
#     cv2.destroyAllWindows()