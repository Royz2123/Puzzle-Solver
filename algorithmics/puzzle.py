import cv2
import numpy as np
import copy

from constants import *
import algorithmics.connect_puzzle as connect_puzzle


class Puzzle(object):
    # shape and color
    METHODS = [0, 1]

    def __init__(self, pieces):
        self._pieces = pieces

        self._corners = [piece for piece in pieces if piece.is_puzzle_corner()]
        self._edges = [piece for piece in pieces if piece.is_puzzle_edge()]
        self._regulars = [
            piece for piece in pieces
            if not piece.is_puzzle_edge() and not piece.is_puzzle_corner()
        ]

        self._width, self._height = self.get_dimensions()

        # check if puzzle detected is good
        if len(self._corners) != 4:  # or len(self._edges) != (self._width + self._height - 2 - 4) * 2:
            raise Exception(
                "Puzzle parsed wrong:\nCorners: %d/4\nWidth: %s\nHeight: %s\n" % (
                    len(self._corners),
                    self._width,
                    self._height
                )
            )
        else:
            print("Connecting puzzle of dimensions: %s x %s" % (self._width, self._height))

        # maybe other way round? matters?
        self._final_puzzle = []
        self._connected_puzzle = []

    def get_pieces(self):
        return self._pieces

    def get_dimensions(self):
        total_edges = len(self._edges) + 2 * len(self._corners)

        p4 = total_edges / 4
        delta = np.sqrt(p4 ** 2 - len(self._pieces))

        return p4 + delta, p4 - delta

    def display(self):
        mat = []
        for row in self._final_puzzle:
            row = [cv2.resize(piece.get_rotated_piece(edge).copy(), (200, 200)) for piece, edge in row]
            row_pic = np.concatenate(tuple(row))
            mat.append(row_pic)

        img = np.concatenate(tuple(mat), axis=1)

        cv2.imshow("Puzzle", img)
        # cv2.waitKey(0)

    # TODO: consider pieces from other rows
    def complete_row_old(self, first_piece, first_edge, curr_pieces, border, row_length=None):
        row = [(first_piece, (first_edge + 2) % 4)]
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

            if not len(supply):
                supply = curr_pieces

            curr_piece, connector_edge = self.find_closest_piece_edge(
                curr_piece,
                curr_edge,
                supply
            )
            row.append((curr_piece, connector_edge))
            curr_pieces.remove(curr_piece)

            # have we finished?
            if (
                    curr_piece.is_puzzle_corner()
                    or (not border and curr_piece.is_puzzle_edge())
                    or len(row) == row_length
            ):
                break

            # otherwise
            curr_edge = (connector_edge + 2) % 4
        return row

    def greedy_old(self):
        curr_pieces = copy.copy(self._pieces)

        # start with corner
        first_piece = [p for p in curr_pieces if p.is_puzzle_corner()][2]
        first_edges = first_piece.get_puzzle_regs_indices()

        # move along first indices
        first_row = self.complete_row_old(first_piece, first_edges[0], curr_pieces, True)
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
                row_puzzle_edge = row_first_piece.get_puzzle_edges_indices()[0]

            else:
                row_first_piece, row_connecting_edge = self.find_closest_piece_edge(
                    row_first_piece,
                    row_connecting_edge,
                    [p for p in curr_pieces if p.is_puzzle_corner()]
                )
                row_puzzle_edge = (row_connecting_edge - 1) % 4

            row_puzzle_edge = (row_puzzle_edge + 2) % 4
            row_connecting_edge = (row_connecting_edge + 2) % 4

            # find the row
            curr_row = self.complete_row_old(row_first_piece, row_puzzle_edge, curr_pieces, last_row, row_length)
            self._final_puzzle.append(curr_row)

            # print("First piece", row_first_piece)

        # flip puzzle if wrong, not sure about this
        if (first_edges[1] - first_edges[0]) % 4 == 1:
            self._final_puzzle = self._final_puzzle[::-1]

    def connect(self):
        self._connected_puzzle, connected_images = connect_puzzle.get_solved_puzzle_img(self._final_puzzle)

        for index, image in enumerate(connected_images):
            cv2.imshow("big pic", cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
            cv2.imwrite(".\\image_processing\\results\\videos\\%d.png" % index, image)
            cv2.waitKey(100)
        cv2.waitKey(0)

        # print data from the connected puzzle
        img = connected_images[-1]
        for row in self._connected_puzzle:
            for piece, center, angle in row:
                cv2.circle(img, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
        # cv2.imshow("mat", img)
        # cv2.waitKey(0)

    # gets list of pieces and chooses best overall method
    def choose_best_method(self, pcs_dst_methods):
        rankings = []
        for i, pcs_dst in enumerate(pcs_dst_methods):
            rankings.append([])

            # sort and add rankings
            pcs_dst.sort(key=lambda x: x[2])
            for j, triple in enumerate(pcs_dst):
                rankings[i].append((triple[0], triple[1], j))

        # sort by piece index
        for i in range(len(rankings)):
            # print("Rankings i: ", rankings[i])
            rankings[i].sort(key=lambda x: x[1] + 4 * x[0].get_index())
            # print("Rankings i: ", rankings[i])

        # merge arrays (Assume they all have the same edges)
        final_ranking = []
        for i in range(len(rankings[0])):
            # Maybe weighted sum ?
            score = sum([ranking[i][2] for ranking in rankings])
            final_ranking.append((rankings[0][i][0], rankings[0][i][1], score))

        final_ranking.sort(key=lambda x: x[2])
        # print("Final Rankings: ", final_ranking)

        # get minimal score
        return min(final_ranking, key=lambda x: x[2])

    def create_command_list(self):
        commands = []

        for ridx, row in enumerate(self._connected_puzzle):
            for piece, center, angle in row:
                angle = (angle - np.pi / 2) % (2 * np.pi)
                commands.append((
                    1793 - piece.get_pickup()[0],
                    piece.get_pickup()[1],
                    FIRST_POS[0] + center[0],
                    FIRST_POS[1] + center[1],
                    angle if angle <= np.pi else (angle - 2 * np.pi),
                ))
        return commands

    # TODO: consider pieces from other rows
    def complete_row(self, first_piece, first_edge, curr_pieces, border, row_length=None, row_before=None):
        row = [(first_piece, (first_edge + 2) % 4)]
        curr_pieces.remove(first_piece)  # makes sense

        if not len(curr_pieces):
            return row

        piece_index = 1
        curr_piece, curr_edge = first_piece, first_edge
        prev_piece, prev_edge = first_piece, first_edge

        while True:
            if row_length is None:
                supply = [p for p in curr_pieces if p.is_puzzle_corner() or p.is_puzzle_edge()]
            elif border:
                if len(row) == (row_length - 1):
                    supply = [p for p in curr_pieces if p.is_puzzle_corner()]
                else:
                    supply = [p for p in curr_pieces if p.is_puzzle_edge()]
            else:
                if len(row) == (row_length - 1):
                    supply = [p for p in curr_pieces if p.is_puzzle_edge()]
                else:
                    supply = [p for p in curr_pieces if not p.is_puzzle_edge() and not p.is_puzzle_corner()]

            # if supply is empty
            if not len(supply):
                supply = curr_pieces

            # find next piece
            pcs_dists_methods = []
            for method in Puzzle.METHODS:
                if row_before is None:
                    pcs_dists = [(x, curr_piece.compare_edge_to_piece(curr_edge, x, method)) for x in supply]

                    # set in format that we want
                    pcs_dists = [(piece, edge, score) for piece, data in pcs_dists for edge, score in data]
                    pcs_dists.sort(key=lambda x: x[2])

                    pcs_dists = [
                        triple for triple in pcs_dists
                        if (triple[1] + 1) % 4 in triple[0].get_puzzle_edges_indices()
                    ]

                if not row_before is None and len(row_before) - 1 >= piece_index:
                    pcs_dists_1 = [(x, curr_piece.compare_edge_to_piece(curr_edge, x, method)) for x in supply]
                    pcs_dists_2 = [
                        (x, row_before[piece_index][0].compare_edge_to_piece((row_before[piece_index][1] - 1) % 4, x,
                                                                             method)) for x
                        in supply]

                    pcs_dists = []
                    for i in range(len(pcs_dists_1)):
                        piece1, data1 = pcs_dists_1[i]
                        _, data2 = pcs_dists_2[i]

                        # create new dataset
                        for edge1, score1 in data1:
                            find_edge = (edge1 + 1) % 4
                            nexts = [(edge2, score2) for edge2, score2 in data2 if edge2 == find_edge]
                            if len(nexts):
                                pcs_dists.append((piece1, edge1, score1 + nexts[0][1]))

                pcs_dists_methods.append(pcs_dists)

            if not len(pcs_dists_methods[0]):
                return row, True
            curr_piece, connector_edge, score = self.choose_best_method(pcs_dists_methods)

            try:
                print("Found Piece! %s is connected to %s by (%d, %d)" % (
                    curr_piece,
                    prev_piece,
                    connector_edge,
                    prev_edge
                ))

                row.append((curr_piece, connector_edge))
                curr_pieces.remove(curr_piece)
            except:
                print("Failed on Row")
                return row, True

            prev_piece = curr_piece
            prev_edge = (connector_edge + 2) % 4

            # have we finished?
            if (
                curr_piece.is_puzzle_corner()
                or (not border and curr_piece.is_puzzle_edge())
                or len(row) == row_length
            ):
                break

            # otherwise
            curr_edge = (connector_edge + 2) % 4
            piece_index += 1
        return row, False

    def find_closest_piece_edge(self, p, idx, pcs):
        pcs_dists_methods = []
        for method in Puzzle.METHODS:
            pcs_dists = [(x, p.compare_edge_to_piece(idx, x, method)) for x in pcs]
            pcs_dists = [(piece, edge, score) for piece, data in pcs_dists for edge, score in data]
            pcs_dists.sort(key=lambda x: x[2])

            if not len(pcs_dists):
                print("No fitting piece")

            pcs_dists_methods.append(pcs_dists)
        curr_piece, connector_edge, score = self.choose_best_method(pcs_dists_methods)
        return curr_piece, connector_edge

    def greedy(self):
        corners = [p for p in self._pieces if p.is_puzzle_corner()]
        for corner in corners[1:]:
            self._final_puzzle = []
            if self.greedy_try_corner(corner):
                break
            # print(e)
            # print("Corner failed: %s" % corner)

    def greedy_try_corner(self, first_piece):
        curr_pieces = copy.copy(self._pieces)

        # start with corner
        first_edges = first_piece.get_puzzle_edges_indices()

        # get first piece orientation
        edge1, edge2 = tuple(first_edges)
        if edge2 == 0:
            print("Bad Corner")
            return False

        # move along first indices
        first_row, stop = self.complete_row(first_piece, (edge1 + 2) % 4, curr_pieces, True)
        self._final_puzzle.append(first_row)
        row_length = len(first_row)

        row_first_piece = first_piece
        row_connecting_edge = (edge2 + 2) % 4
        curr_row = first_row

        while len(curr_pieces) > 0:
            last_row = (len(curr_pieces) <= row_length)

            # find first piece in row connect_puzzle
            if not last_row:
                supply = [p for p in curr_pieces if p.is_puzzle_edge()]
            else:
                supply = [p for p in curr_pieces if p.is_puzzle_corner()]

            if not len(supply):
                supply = curr_pieces

            print("Moving to next row: Finding Partner for: %s edge %d" % (
                row_first_piece._name,
                row_connecting_edge
            ))

            bad_future = []
            not_found = False
            flag = True
            while flag:
                if not len(supply):
                    break

                row_first_piece_temp, row_connecting_edge_temp = self.find_closest_piece_edge(
                    row_first_piece,
                    row_connecting_edge,
                    supply
                )

                flag = not ((row_connecting_edge_temp - 1) % 4) in row_first_piece_temp.get_puzzle_edges_indices()
                if flag:
                    supply.remove(row_first_piece_temp)
                    bad_future.append(row_first_piece_temp)
                else:
                    row_first_piece = row_first_piece_temp
                    row_connecting_edge = row_connecting_edge_temp

            supply += bad_future
            row_puzzle_edge = (row_connecting_edge + 1) % 4
            row_connecting_edge = (row_connecting_edge + 2) % 4

            # find the row
            curr_row, stop = self.complete_row(row_first_piece, row_puzzle_edge, curr_pieces, last_row,
                                               row_length, curr_row)
            self._final_puzzle.append(curr_row)

            if stop:
                break

        # flip puzzle, not sure about this
        if len(curr_pieces):
            return False
        return True
