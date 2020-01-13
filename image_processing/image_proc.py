import numpy as np
import cv2

import util
from constants import *
import piece
import parse_pieces

def get_pieces(test=False):
    if test:
        above, below = util.get_test_images()
    else:
        # TODO: Take image
        above, below = (None, None)

    below, binary = parse_pieces.create_mask(below)
    masked = parse_pieces.mask_rgb(above, binary)

    util.output("below", below)
    cv2.waitKey(0)

    not_masked = parse_pieces.show_diffs(above, binary)
    recoged, pieces = parse_pieces.recog_pieces(masked, below, binary)

    pieces[0].compare_piece(pieces[1])

    return pieces

    # pieces[0].compare_shape(pieces[1])
    #
    # # util.output('above masked', masked)
    # # util.output('above masked with centers', recoged)
    #
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    get_pieces(test=True)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()