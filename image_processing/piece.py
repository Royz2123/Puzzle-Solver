import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
from scipy import ndimage
import itertools

from constants import *
import test
import util

class Piece:

    def __init__(self, above, below, index):
        self._above = above
        self._below = below
        self._index = index
        self._name = "Piece: %d" % index
        self._display = above.copy()

        self._centroid = self.find_centroid()
        self._raw_color_edges, self._raw_real_edges = self.find_edges()

        self._corners, self._corner_angles = self.find_corners()
        self._color_edges = self.divide_edges(self._raw_color_edges)
        self._real_edges = self.divide_edges(self._raw_real_edges)

        self._edge_images = self.create_shape_vector()
        self._color_vectors = self.create_color_vector()


    def find_centroid(self):
        # connected compnents
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(self._below)

        if len(centroids) > 2:
            print("More than one center recoged for piece")

        index = 1 # first index will be for black part, second for actual piece
        centroid = tuple(list(centroids[index].astype(int)))
        cv2.circle(self._display, centroid, 3, (0, 0, 255), 7)

        return centroid

    def display_piece(self):
        edges = np.concatenate(tuple(self._edge_images), axis=0)
        general = np.concatenate((
            self._display,
            self._above,
            cv2.cvtColor(self._below, cv2.COLOR_GRAY2RGB)
        ))

        cv2.imshow(self._name + "_edges", edges)
        cv2.imshow(self._name, general)

    def find_corners(self):
        # Method 1
        corners1 = cv2.goodFeaturesToTrack(self._below, 10, 0.1, 10)
        corners1 = np.int0(corners1)

        # Method 2
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

        #
        peaks, _ = find_peaks(dists, prominence=(10), threshold=(0, 2))
        plt.plot(angles[peaks], dists[peaks], "x")
        # plt.plot(angles, dists)
        # plt.show()
        plt.clf()

        possible_corners = list(zip(peaks, angles[peaks]))
        pairs = list(itertools.combinations(possible_corners, 2))
        pairs = [(a1[0], a2[0], abs(a1[1] - a2[1])) for a1, a2 in pairs]
        top_pairs = sorted(pairs, key=lambda x: abs(x[2] - np.pi/2))[:3]

        corners2 = list(set(
            [pair[0] for pair in top_pairs]
            + [pair[1] for pair in top_pairs]
        ))
        corners2 = points[corners2].tolist()
        corners2.sort(key=lambda x:
            math.atan2(x[1] - self._centroid[1], x[0] - self._centroid[0])
        )


        for i in corners1:
            x, y = i.ravel()
            cv2.circle(self._display, (x, y), 5, color=(255, 255, 0), thickness=-1)

        for index, i in enumerate(corners2):
            x, y = i
            cv2.circle(self._display, (x, y), 3, color=(255, 0, 0), thickness=-1)
            cv2.putText(self._display, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

        corner_angles = [
            (math.atan2(point[1] - self._centroid[1], point[0] - self._centroid[0]))
            for point in corners2
        ]
        return corners2, corner_angles

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
        # values = self._above[self._color_edges]
        for edge_class in self._color_edges:
            plt.plot(edge_class)
            plt.show()

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
            middle = 100
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

            edge_images.append(im_floodfill.astype(np.uint8))

        return edge_images

    def compare_shape(self, other):
        scores = []
        for idx1, edge_image_1 in enumerate(self._edge_images):
            for idx2, edge_image_2 in enumerate(other._edge_images):
                edge_image_1[edge_image_1 > 1] = 1
                edge_image_2[edge_image_2 > 1] = 1

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

                xored = cv2.bitwise_xor(frame1, frame2)
                xored = cv2.bitwise_not(xored)
                score = np.sum(xored)

                # xored = (frame1 + frame2) * 100
                # cv2.imshow("XOR", xored)
                # cv2.waitKey(0)

                scores.append((idx1, idx2, score))

        scores.sort(key=lambda x: x[2])
        print(scores[:2])



