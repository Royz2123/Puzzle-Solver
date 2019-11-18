import numpy as np
import cv2

BASE = "./test_images/"
ABOVE = BASE + "above.jpeg"
BELOW = BASE + "below.jpeg"

RESULTS_BASE = "./results/"

THRESH = 75
CNT_THRESH = 50

def relevant_section(img):
    return img[-900:-200,-650:-200]

def output(name, img):
    cv2.imshow(name, img)
    cv2.imwrite(RESULTS_BASE + name + ".png", img)


above = cv2.imread(ABOVE,1)
below = cv2.imread(BELOW,0)

above = relevant_section(above)
below = relevant_section(below)

output('below raw', below)
output('above raw', above)

# thresh below

below[below >= THRESH] = 0
below[below >= 1] = 255

output('below mask', below)

binary = below.copy()
binary[binary >= 1] = 1
mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
masked = above * mask

output('above masked', masked)

not_binary = 1 - binary
mask = cv2.cvtColor(not_binary, cv2.COLOR_GRAY2RGB)
not_masked = above * mask

output('not masked', not_masked)

recoged = masked.copy()

# connected compnents
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(below)

# contours
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]

for i in range(1, len(centroids)):
    cv2.circle(recoged, tuple(list(centroids[i].astype(int))), 3, (0, 0, 255), 7)

for cnt in contours:
    if cv2.contourArea(cnt) > CNT_THRESH:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(recoged, (x, y), (x + w, y + h), (0, 255, 0), 2)

output('above masked with centers', recoged)

k = cv2.waitKey(0)
cv2.destroyAllWindows()

# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()

#
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',img)
#     cv2.destroyAllWindows()