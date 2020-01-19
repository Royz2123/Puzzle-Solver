import numpy as np
import cv2

import image_processing.util as util
from constants import *
import image_processing.piece as piece


def create_mask(below, test):
    if test:
        below[below >= TEST_THRESH] = 0
    else:
        below[below >= REAL_THRESH] = 0

    below[below >= 1] = 255

    kernel = np.ones((3, 3), np.uint8)
    below = cv2.morphologyEx(below, cv2.MORPH_OPEN, kernel)
    below = cv2.morphologyEx(below, cv2.MORPH_CLOSE, kernel)

    binary = below.copy()
    binary[binary >= 1] = 1

    return below, binary


def mask_rgb(above, binary):
    mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    masked = above * mask
    return masked


def show_diffs(above, binary):
    not_binary = 1 - binary
    mask = cv2.cvtColor(not_binary, cv2.COLOR_GRAY2RGB)
    not_masked = above * mask

    return not_masked


def recog_pieces(above, below, binary):
    recoged = above.copy()
    pieces = []

    # contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    contours = [(c, cv2.contourArea(c)) for c in contours]

    # remove small contours
    contours = [c for c in contours if c[1] > CNT_THRESH]

    # has big contours
    smallest = min(contours, key=lambda x: x[1])
    overlapping = [c for c in contours if c[1] > smallest[1]*1.5]

    # return if has overlapping
    if len(overlapping):
        for cnt, area in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if (cnt, area) not in overlapping:
                cv2.rectangle(recoged, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(recoged, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(recoged, "Overlapping:", (x, y-15), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255))
        return recoged, None

    index = 0
    for cnt, area in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(recoged, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pieces.append(piece.Piece(
            above[y - PIECE_MARGIN : y + h + PIECE_MARGIN, x - PIECE_MARGIN : x + w + PIECE_MARGIN].copy(),
            below[y - PIECE_MARGIN : y + h + PIECE_MARGIN, x - PIECE_MARGIN : x + w + PIECE_MARGIN].copy(),
            index,
            np.array([x - PIECE_MARGIN, y - PIECE_MARGIN])
        ))
        pieces[index].display_piece()

        # plot on original piece
        centroid = tuple(pieces[index].get_real_centroid().tolist())
        cv2.circle(recoged, centroid,  15, [0, 0, 255], -1)
        for corner in pieces[index].get_real_corners():
            cv2.circle(recoged, tuple(corner.tolist()), 15, [255, 0, 0], -1)

        index += 1

    return recoged, pieces