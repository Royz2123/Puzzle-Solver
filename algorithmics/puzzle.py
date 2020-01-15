import cv2
import numpy as np

import copy


class Puzzle(object):
    def __init__(self, pieces):
        self._pieces = pieces

        self._corners = [piece for piece in pieces if piece.is_puzzle_corner()]
        self._edges = [piece for piece in pieces if piece.is_puzzle_edge()]
        self._regulars = [
            piece for piece in pieces
            if not piece.is_puzzle_edge() and not piece.is_puzzle_corner()
        ]

        self._width, self._height = self.get_dimensions()

        # maybe other way round? matters?
        self._final_puzzle = []

    def get_dimensions(self):
        total_edges = len(self._edges) + 2 * len(self._corners)

        p4 = total_edges / 4
        delta = np.sqrt(p4 ** 2 - len(self._pieces))

        return p4 + delta, p4 - delta

    def find_closest_piece_edge(self, p, idx, pcs):
        pcs_dists = [(x, p.compare_piece_edge(idx, x)) for x in pcs]
        piece, edge_scores = min(pcs_dists, key=lambda x: x[1][0][1])
        print(piece, edge_scores[0][0])
        return piece, edge_scores[0][0]

    def display(self):
        mat = []
        for row in self._final_puzzle:
            row = [cv2.resize(piece.get_rotated_piece(edge).copy(), (200, 200)) for piece, edge in row]
            row_pic = np.concatenate(tuple(row))
            mat.append(row_pic)
        img = np.concatenate(tuple(mat), axis=1)

        cv2.imshow("Puzzle", img)
        cv2.waitKey(0)


    # TODO: consider pieces from other rows
    def complete_row(self, first_piece, first_edge, curr_pieces, border, row_length=None):
        row = [(first_piece, (-first_edge) % 4)]
        curr_pieces.remove(first_piece)

        curr_edge = first_edge
        curr_piece = first_piece
        while True:
            if border:
                supply = [p for p in curr_pieces if p.is_puzzle_corner() or p.is_puzzle_edge()]
            elif row_length is not None and (len(row) == (row_length - 1)):
                supply = [p for p in curr_pieces if p.is_puzzle_edge()]
            else:
                supply = curr_pieces

            curr_piece, connector_edge = self.find_closest_piece_edge(
                curr_piece,
                curr_edge,
                supply
            )
            row.append((curr_piece, curr_edge))
            curr_pieces.remove(curr_piece)

            # have we finished?
            if (
                curr_piece.is_puzzle_corner()
                or (not border and curr_piece.is_puzzle_edge())
            ):
                break

            # otherwise
            curr_edge = (connector_edge + 2) % 4
        return row

    def greedy(self):
        curr_pieces = copy.copy(self._pieces)

        # start with corner
        first_piece = [p for p in curr_pieces if p.is_puzzle_corner()][2]
        first_edges = first_piece.get_puzzle_regs_indices()

        # move along first indices
        first_row = self.complete_row(first_piece, first_edges[0], curr_pieces, True)
        self._final_puzzle.append(first_row)
        row_length = len(first_row)

        row_first_piece = first_piece
        row_connecting_edge = first_edges[1]

        while len(curr_pieces) > 0:
            last_row = (len(curr_pieces) <= row_length)

            # find first piece in row
            if not last_row:
                row_first_piece, row_connecting_edge = self.find_closest_piece_edge(
                    row_first_piece,
                    row_connecting_edge,
                    [p for p in curr_pieces if p.is_puzzle_edge()]
                )
            else:
                row_first_piece, row_connecting_edge = self.find_closest_piece_edge(
                    row_first_piece,
                    row_connecting_edge,
                    [p for p in curr_pieces if p.is_puzzle_corner()]
                )

            row_puzzle_edge = row_first_piece.get_puzzle_edges_indices()[0]
            row_puzzle_edge = (row_puzzle_edge + 2) % 4
            row_connecting_edge = (row_connecting_edge + 2) % 4

            # find the row
            curr_row = self.complete_row(row_first_piece, row_puzzle_edge, curr_pieces, last_row, row_length)
            self._final_puzzle.append(curr_row)

            print("First piece", row_first_piece)



