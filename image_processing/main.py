import numpy as np
import cv2

import util
from constants import *
import piece
import parse_pieces

above, below = util.get_test_images()
below, binary = parse_pieces.create_mask(below)
masked = parse_pieces.mask_rgb(above, binary)

not_masked = parse_pieces.show_diffs(above, binary)
recoged, pieces = parse_pieces.recog_pieces(masked, below, binary)

pieces[0].compare_shape(pieces[1])

# util.output('above masked', masked)
# util.output('above masked with centers', recoged)

k = cv2.waitKey(0)
cv2.destroyAllWindows()