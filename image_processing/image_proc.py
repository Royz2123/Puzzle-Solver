import numpy as np
import cv2

import image_processing.util as util
from constants import *
import image_processing.piece as piece
import image_processing.parse_pieces as parse_pieces


def get_pieces(above, below, test):
    below, binary = parse_pieces.create_mask(below, test)

    util.output("below", binary * 255)
    cv2.waitKey(0)

    masked = parse_pieces.mask_rgb(above, binary)

    util.output("below", below)
    cv2.waitKey(0)

    not_masked = parse_pieces.show_diffs(above, binary)

    util.output("not masked", not_masked)
    cv2.waitKey(0)

    recoged, pieces, overlapping = parse_pieces.recog_pieces(masked, below, binary)

    util.output("test", recoged)
    cv2.waitKey(0)

    if overlapping:
        print("Overlapping Pieces!")
        location = parse_pieces.find_empty_place(binary)
        return [pieces, location], False
    else:
        pieces[0].compare_piece_to_piece(pieces[1], 0)
        return pieces, True

    # pieces[0].compare_shape(pieces[1])
    #
    # # util.output('above masked', masked)
    # # util.output('above masked with centers', recoged)
    #
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    get_pieces(test=True)

    # k = cv2.waitKey(0)
    cv2.destroyAllWindows()