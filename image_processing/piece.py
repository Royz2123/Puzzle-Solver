import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy.signal import correlate
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import peak_prominences
from scipy import ndimage
import itertools

import image_processing.util as util
from constants import *


class Piece(object):
    def __init__(self, above, below, index, relative_pos):
        self._relative_pos = relative_pos
        self._above = above
        self._below = below
        self._index = index
        self._name = "Piece: %d" % index

        self.remove_non_piece()

        self._display = above.copy()

        self._centroid = self.find_centroid()
        self._theta = 0
        self._raw_color_edges, self._raw_real_edges = self.find_edges()

        self._corners, self._corner_angles = self.find_corners()
        self._color_edges = self.divide_edges(self._raw_color_edges)
        self._real_edges = self.divide_edges(self._raw_real_edges)

        self._edge_images = self.create_shape_vector()
        self._color_vectors = self.create_color_vector()

        self._puzzle_edges = self.puzzle_edges()


    def __repr__(self):
        return self._name

    def get_real_centroid(self):
        return self._centroid + self._relative_pos

    def get_index(self):
        return self._index

    def get_real_corners(self):
        return [np.array(corner) + self._relative_pos for corner in self._corners]

    def display_color_edge(self, color_vector):
        cv2.imshow("colors", np.repeat(np.array([color_vector]), 20, axis=0))
        cv2.waitKey(0)

    def display_color_comparison(self, color_vector1, color_vector2):
        img1 = np.repeat(np.array([color_vector1]), 20, axis=0)
        img2 = np.repeat(np.array([color_vector2]), 20, axis=0)
        cv2.imshow("colors", np.concatenate((img1, img2), axis=0))
        cv2.waitKey(0)

    def get_theta(self):
        return self._theta

    def get_centroid(self):
        return self._centroid

    def get_rotated_piece(self, edge):
        self._theta = self._corner_angles[edge] + 3 * np.pi / 4
        return ndimage.rotate(self._display, self._theta * 180 / np.pi)

    def remove_non_piece(self):
        contours, _ = cv2.findContours(self._below, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_c = max(contours, key=cv2.contourArea)

        for c in contours:
            if len(c) < len(max_c):
                cv2.drawContours(self._above, [c], -1, (0, 0, 0), thickness=cv2.FILLED)
                cv2.drawContours(self._below, [c], -1, 0, thickness=cv2.FILLED)

    def find_centroid(self):
        # connected compnents
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._below)

        if len(centroids) > 2:
            print("More than one center recoged for piece")

        index = 1 # first index will be for black part, second for actual piece
        centroid = tuple(list(centroids[index].astype(int)))
        cv2.circle(self._display, centroid, 3, (0, 0, 255), 7)
        cv2.putText(self._display, str(self._index), centroid, cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))
        return centroid

    def display_piece(self):
        edges = np.concatenate(tuple([img*255 for img in self._edge_images]), axis=0)
        general = np.concatenate((
            self._display,
            self._above,
            cv2.cvtColor(self._below, cv2.COLOR_GRAY2RGB)
        ))

        cv2.imshow(self._name + "_edges", cv2.resize(edges, dsize=(100, 400)))
        cv2.imshow(self._name, general)

        cv2.imwrite(PIECES_BASE + self._name.replace(":", "") + ".png", general)
        cv2.imwrite(PIECES_BASE + self._name.replace(":", "") + " - edges.png", edges)

    def find_corners(self, method=3):
        # Method 1
        if method == 1:
            corners = cv2.goodFeaturesToTrack(self._below, 10, 0.1, 10)
            corners = np.int0(corners)

        # Method 2
        elif method == 2:
            edge_info = ([(
                point,
                np.linalg.norm(point - np.array(self._centroid)),
                math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0])
            ) for index, point in enumerate(self._raw_real_edges)])
            edge_info.sort(key=lambda x: x[2])

            points = np.array([edge[0] for edge in edge_info])
            dists = np.array([edge[1] for edge in edge_info])
            angles = np.array([edge[2] for edge in edge_info])

            # dervs = np.gradient(dists)
            # dervs[abs(dervs) > 2] = 0

            length = len(dists)
            points = np.concatenate((points, points[length//16:]))
            dists = np.lib.pad(dists, (0, length // 16), 'wrap')
            angles = np.lib.pad(angles, (0, length // 16), 'wrap')
            angles[length:] += 2 * np.pi

            peaks, _ = find_peaks(dists, prominence=(5), threshold=(0, 3), width=(15))
            prominences, left_bases, right_bases = peak_prominences(dists, peaks)
            offset = np.ones_like(prominences) * 2
            widths = peak_widths(
                dists, peaks,
                rel_height=1,
                prominence_data=(offset, left_bases, right_bases)
            )
            pairs = sorted(list(zip(peaks, widths[0])), key=lambda x: x[1])[:5]
            peaks = np.array([pair[0] for pair in pairs])

            # print(list(zip(peaks, widths)))
            # print(widths)

            angles[length:] -= 2 * np.pi
            peaks %= length

            plt.plot(peaks, dists[peaks], "x")
            plt.hlines(*widths[1:], color="C2")
            plt.plot(dists)
            # plt.show()

            plt.savefig(".\\image_processing\\pieces\\%s_corner" % (
                self._name.replace(" ", "_")
            ))
            plt.clf()

            # remove pairs that are the same
            peaks = list(set(peaks))
            if len(set(peaks)) == 5:
                peaks = peaks[:4]
            corners = peaks
            corners.sort(
                key=lambda x:
                math.atan2(x[1] - self._centroid[1], x[0] - self._centroid[0])
            )

        elif method == 3:
            edge_info = ([(
                point,
                np.linalg.norm(point - np.array(self._centroid)),
                math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0])
            ) for index, point in enumerate(self._raw_real_edges)])
            edge_info.sort(key=lambda x: x[2])

            points = np.array([edge[0] for edge in edge_info])
            dists = np.array([edge[1] for edge in edge_info])
            angles = np.array([edge[2] for edge in edge_info])

            # dervs = np.gradient(dists)
            # dervs[abs(dervs) > 2] = 0

            length = len(dists)
            points = np.concatenate((points, points[length // 16:]))
            dists = np.lib.pad(dists, (0, length // 16), 'wrap')
            angles = np.lib.pad(angles, (0, length // 16), 'wrap')
            angles[length:] += 2 * np.pi

            peaks, _ = find_peaks(dists, prominence=(5), threshold=(0, 3), width=(10))
            plt.plot(angles[peaks], dists[peaks], "x")
            plt.plot(angles, dists)
            # plt.plot(dists[peaks], "x")
            # plt.plot(dists)
            plt.savefig(".\\image_processing\\pieces\\%s_corner" % (
                self._name.replace(" ", "_").replace(":", "")
            ))
            plt.clf()

            # angles = np.array([angle - 2 * np.pi for angle in angles])
            angles[length:] -= 2 * np.pi
            possible_corners = list(zip(peaks, angles[peaks]))
            pairs = list(itertools.combinations(possible_corners, 2))
            pairs = [(a1[0] % length, a2[0] % length, abs(a1[1] - a2[1])) for a1, a2 in pairs]

            # sort pairs
            pairs = [(a1, a2, diff) if a1 < a2 else (a2, a1, diff) for a1, a2, diff in pairs]

            # remove pairs that are the same
            equivalents = [pair for pair in pairs if pair[2] == 0]
            to_remove = [pair[1] for pair in equivalents]
            pairs = [pair for pair in pairs if pair[1] not in to_remove]

            if len(to_remove):
                print(self._name, ":\tRemoved Pairs: ", to_remove)

            top_pairs = sorted(pairs, key=lambda x: abs(x[2] - np.pi / 2))[:5]
            top_pairs = [x for x in top_pairs if abs(x[2] - np.pi / 2) < 0.3 * (np.pi / 2)]
            top_pairs = sorted(top_pairs, key=lambda x: x[0])

            # try to find chain, otherwise remove
            chains = util.get_chains(top_pairs)
            max_chain = max(chains, key=len)

            if len(max_chain) >= 3:
                top_pairs = max_chain
            else:
                print(self._name, ":\tMax Chain not found! Chains: ", chains)
                top_pairs = top_pairs[:3]

            corners = list(set(
                [pair[0] for pair in top_pairs]
                + [pair[1] for pair in top_pairs]
            ))[:4]

            dists = np.lib.pad(dists[:length], (length // 4, length // 4), 'wrap')
            new_corners = []
            width = 10
            radius = 7
            thresh = length // 20

            for corner in corners:
                ind = corner + length // 4
                new_dists = []
                for i in range(ind - width - radius, ind - radius):
                    if np.abs(dists[ind] - dists[i]) < thresh:
                        new_dists.append([i, dists[i]])
                hat = np.array(new_dists)
                dists1 = hat[:, 1]
                x_axis = hat[:, 0]
                new_dists = []
                for i in range(ind + radius, ind + width + radius):
                    if np.abs(dists[ind] - dists[i]) < thresh:
                        new_dists.append([i, dists[i]])
                hat = np.array(new_dists)
                dists2 = hat[:, 1]
                y_axis = hat[:, 0]

                params_before = np.polyfit(x_axis, dists1, 1, rcond=None, full=False, w=None, cov=False)
                params_after = np.polyfit(y_axis, dists2, 1, rcond=None, full=False, w=None, cov=False)
                theta = -(params_before[1] - params_after[1]) / (params_before[0] - params_after[0])
                new_corners.append(int(np.round(theta)) - length // 4)
            corners = new_corners

            corners = points[corners].tolist()
            corners.sort(
                key=lambda x:
                math.atan2(x[1] - self._centroid[1], x[0] - self._centroid[0])
            )

        for index, i in enumerate(corners):
            x, y = i
            cv2.circle(self._display, (x, y), 3, color=(255, 0, 0), thickness=-1)
            cv2.putText(self._display, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

        corner_angles = [
            (math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0]))
            for point in corners
        ]
        return corners, corner_angles

    def find_edges(self):
        # genius! lets erode the image just by a bit and then definitely get edges on the inside
        kernel = np.ones((COLOR_MARGIN, COLOR_MARGIN), np.uint8)
        eroded = cv2.erode(self._below, kernel)

        color_edges_img = cv2.Canny(eroded, 100, 255)
        color_indices = np.where(color_edges_img != [0])
        color_edges = np.array(list(zip(color_indices[1], color_indices[0])))

        real_edges_img = cv2.Canny(self._below, 100, 255)
        real_indices = np.where(real_edges_img != [0])
        real_edges = np.array(list(zip(real_indices[1], real_indices[0])))

        self._display[color_indices] = [0, 255, 255]
        self._display[real_indices] = [255, 255, 0]

        return color_edges, real_edges

    def divide_edges(self, edges):
        edge_angles = [
            ((math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0])), point)
            for point in edges
        ]
        edge_angles.sort(key=lambda x: x[0])

        divided_edges = [[]]
        corner_angles = self._corner_angles + [np.inf]
        curr_angle_index = 0
        curr_edges = []

        for angle, edge in edge_angles:
            if angle > corner_angles[curr_angle_index]:
                curr_angle_index += 1
                divided_edges.append([])
            divided_edges[curr_angle_index].append(edge)

        divided_edges[4] += divided_edges[0]
        divided_edges = divided_edges[1:]

        for index, edge_class in enumerate(divided_edges):
            for edge in edge_class:
                x, y = tuple(edge)
                cv2.circle(self._display, (x, y), 2, color=(50*index + 50, 100, 50*index + 50), thickness=-1)

        return divided_edges

    def create_color_vector(self):
        color_vectors = []

        # sort edges by curve
        for color_edge in self._color_edges:
            color_edges_curve = self.make_curve(np.array(color_edge))

            x_s = [edge[1] for edge in color_edges_curve]
            y_s = [edge[0] for edge in color_edges_curve]
            values = self._above[(x_s, y_s)]
            color_vectors.append(values)

            # checks that goldners and prosaks code works
            # for i in range(len(x_s)):
            #     cv2.circle(self._above, (y_s[i], x_s[i]), 3, [255, i // 2, i//2], -1)
            # cv2.imshow("colors", self._above)
            # cv2.waitKey(0)
            # self.display_color_edge(values)

        return color_vectors


    def create_shape_vector(self):
        # for index, angle in enumerate(self._corner_angles):
        #      img = ndimage.rotate(self._display, (angle + 3 * np.pi / 4) * 180 / np.pi)
        #      cv2.imshow(str(index), img)

        edge_images = []
        for index, corner in enumerate(self._corners):
            other_corner = np.array(self._corners[(index + 1) % 4])
            other_corner -= np.array(corner)

            normed_edges = self._real_edges[index]
            normed_edges = [edge - np.array(corner) for edge in normed_edges]

            normed_dists = [util.dist_from_line(other_corner, edge) for edge in normed_edges]

            r_squared = [edge[0]**2 + edge[1] ** 2 for edge in normed_edges]
            normed_xs = [np.sqrt(r2 - dist**2) for r2, dist in zip(r_squared, normed_dists)]

            # middle = int(max(max(normed_dists), abs(min(normed_dists))))
            middle = 200
            shape = (int(max(normed_xs)) + 1, 2*middle + 1)

            mat = np.zeros(shape)

            for x, y in zip(normed_xs, normed_dists):
                mat[int(x)][int(y) + middle] = 255
                mat = mat.astype(np.uint8)

            kernel = np.ones((4, 4), np.uint8)
            mat = cv2.dilate(mat, kernel)

            im_floodfill = mat.copy()

            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = mat.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)

            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0, 0), 255)

            # erode back
            kernel = np.ones((4, 4), np.uint8)
            im_floodfill = cv2.erode(im_floodfill, kernel)
            im_floodfill[im_floodfill > 1] = 1

            # cv2.imshow("mat", im_floodfill*255)
            # cv2.waitKey(0)

            edge_images.append(im_floodfill.astype(np.uint8))

        return edge_images

    def puzzle_edges(self):
        puzzle_edges = []

        for edge_image in self._edge_images:
            frame = np.zeros(edge_image.shape, np.uint8)
            frame[:, :edge_image.shape[1] // 2] = 1

            xored = np.bitwise_xor(frame, edge_image)
            # cv2.imshow("yo", xored * 255)
            # cv2.waitKey(0)
            score = np.sum(xored)

            # print("Shape: ", frame.shape[0])
            # print("Score: ", score)
            puzzle_edges.append(score < frame.shape[0] * 5)

        return puzzle_edges

    def is_puzzle_edge(self):
        return sum(self._puzzle_edges) == 1

    def is_puzzle_corner(self):
        return sum(self._puzzle_edges) > 1

    def get_puzzle_edges_indices(self):
        return [idx for idx in range(len(self._puzzle_edges)) if self._puzzle_edges[idx]]

    def get_puzzle_regs_indices(self):
        return [idx for idx in range(len(self._puzzle_edges)) if not self._puzzle_edges[idx]]


    # COMPARATORS

    def compare_edges_shape(self, idx1, other, idx2):
        edge_image_1 = self._edge_images[idx1]
        edge_image_2 = other._edge_images[idx2]

        new_shape = (max(edge_image_1.shape[0], edge_image_2.shape[0]), edge_image_1.shape[1])
        frame = np.zeros(new_shape, np.uint8)
        frame[:, :edge_image_1.shape[1] // 2] = 1

        frame1 = frame
        frame2 = frame.copy()

        frame1[:edge_image_1.shape[0], :edge_image_1.shape[1]] = edge_image_1
        frame2[:edge_image_2.shape[0], :edge_image_2.shape[1]] = edge_image_2

        # flip because we check matching
        frame2 = cv2.flip(frame2, 1)
        frame2 = cv2.flip(frame2, 0)

        # cv2.imshow("1_" + str(idx1), frame1 * 255)
        # cv2.imshow("2_" + str(idx2), frame2 * 255)

        kernel_erode = np.ones((1, 9), np.uint8)
        kernel_dilate = np.ones((4, 1), np.uint8)
        frame2 = cv2.erode(frame2, kernel_erode, iterations=1)
        frame2 = cv2.dilate(frame2, kernel_dilate, iterations=1)

        xored = cv2.bitwise_xor(frame1, frame2)
        xored = 1 - xored
        score = np.sum(xored)

        xored = (frame1 + frame2) * 100
        # cv2.imshow("XOR", xored * 255)
        # cv2.waitKey(0)

        return score

    def compare_edges_color(self, idx1, other, idx2):
        color_vector_1 = self._color_vectors[idx1]
        color_vector_2 = other._color_vectors[idx2]
        # flip color vector 2
        color_vector_2 = cv2.flip(color_vector_2, 0)
        '''
        :param edge1: (N, 1, 3) array of RGB colors
        :param edge2: (M, 1, 3) array of RGB colors
        :return: best correlation in offset window
        '''
        edge1 = np.reshape(color_vector_1, (color_vector_1.shape[0], 1, color_vector_1.shape[1]))
        edge1 = cv2.cvtColor(edge1, cv2.COLOR_BGR2Lab)
        edge2 = np.reshape(color_vector_2, (color_vector_2.shape[0], 1, color_vector_2.shape[1]))
        edge2 = cv2.cvtColor(edge2, cv2.COLOR_BGR2Lab)
        # make edge1 the longer
        if color_vector_1.shape[0] < color_vector_2.shape[0]:
            edge1, edge2 = edge2, edge1
        # cut off the lightness
        # edge1 = edge1[:, :, 1:]
        # edge2 = edge2[:, :, 1:]

        # normalize and centerize (mean = 0) edges before CC
        nmedge1 = (edge1 - np.mean(edge1, axis=(0, 1), keepdims=True)) / \
                  np.linalg.norm(edge1 - np.mean(edge1, axis=(0, 1), keepdims=True),
                                 axis=(0, 1), keepdims=True)
        nmedge2 = (edge2 - np.mean(edge2, axis=(0, 1), keepdims=True)) / \
                  np.linalg.norm(edge2 - np.mean(edge2, axis=(0, 1), keepdims=True),
                                 axis=(0, 1), keepdims=True)
        # cross correlate
        cc = correlate(nmedge1, nmedge2, 'valid')
        cc_tot = cc.reshape(cc.shape[0])
        score1mean = np.mean(cc_tot)
        # score1max = np.max(cc_tot)  # if len(cc_tot) > 1 else cc_tot[0]
        return -score1mean

    def compare_edges_length(self, idx1, other, idx2):
        return abs(len(self._real_edges[idx1]) - len(other._real_edges[idx2]))

    def compare_edge_to_piece(self, idx1, other, method):
        scores = []
        for idx2 in range(len(self._edge_images)):
            if not self._puzzle_edges[idx1] and not other._puzzle_edges[idx2]:
                score = 0
                if method == 0:
                    score = self.compare_edges_shape(idx1, other, idx2)
                elif method == 1:
                    score = self.compare_edges_color(idx1, other, idx2)
                elif method == 2:
                    score = self.compare_edges_length(idx1, other, idx2)

                scores.append((idx2, score))
        scores.sort(key=lambda x: x[1])
        return scores

    def compare_piece_to_piece(self, other, method):
        scores = []
        for idx1, edge_image_1 in enumerate(self._edge_images):
            curr_scores = self.compare_edge_to_piece(idx1, other, method)
            scores += [(idx1, score[0], score[1]) for score in curr_scores]
        scores.sort(key=lambda x: x[2])
        return scores

    def make_curve(self, cord_array):
        xmax = np.max(cord_array[:, 0])
        ymax = np.max(cord_array[:, 1])
        cord_matrix = np.zeros((xmax + 1, ymax + 1))

        for cord in cord_array:
            cord_matrix[cord[0], cord[1]] = 1

        cord = cord_array[0]
        cnt = 0
        results = np.zeros(cord_array.shape)
        while cnt + 1 < len(cord_array):
            cord_matrix[cord[0], cord[1]] = 0
            min = 10000
            mincord = None

            radius = 1
            while True:
                for i in range(cord[0] - radius, cord[0] + radius + 1):
                    for j in range(cord[1] - radius, cord[1] + radius + 1):
                        if not (i < 0 or i >= len(cord_matrix) or j < 0 or j >= len(cord_matrix[0])):
                            if cord_matrix[i, j] == 1:
                                diq = np.linalg.norm(cord - np.array([i, j]))
                                if diq < min:
                                    min = diq
                                    mincord = np.array([i, j])

                if mincord is None:
                    radius *= 2
                else:
                    break
            results[cnt, :] = cord
            cord = mincord
            cnt += 1
        return results.astype(dtype=np.int)

    def display_real_piece(self):
        big_pic = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE, 3)).astype(dtype=np.uint8)
        r_y, r_x = self._relative_pos
        general = self._above.copy()

        big_pic[r_x: r_x + general.shape[0], r_y: r_y + general.shape[1]] = general

        # centroid = tuple(self.get_real_centroid().tolist())
        # cv2.circle(big_pic, centroid, 10, [0, 0, 255], -1)
        # for corner in self.get_real_corners():
        #     cv2.circle(big_pic, tuple(corner.tolist()), 10, [255, 0, 0], -1)
        return big_pic
