from __future__ import print_function

import numpy as np
import scipy


def get_best_fitting_rect_coords(xy, d_threshold=30, perp_angle_thresh=20, verbose=0):
    """
    Since we expect the 4 puzzle corners to be the corners of a rectangle, here we take
    all detected Harris corners and we find the best corresponding rectangle.
    We perform a recursive search with max depth = 2:
    - At depth 0 we take one of the input point as the first corner of the rectangle
    - At depth 1 we select another input point (with distance from the first point greater
        then d_threshold) as the second point
    - At depth 2 and 3 we take the other points. However, the lines 01-12 and 12-23 should be
        as perpendicular as possible. If the angle formed by these lines is too much far from the
        right angle, we discard the choice.
    - At depth 3, if a valid candidate (4 points that form an almost perpendicular rectangle) is found,
        we add it to the list of candidates.

    Given a list of candidate rectangles, we then select the best one by taking the candidate that maximizes
    the function: area * Gaussian(rectangularness)
    - area: it is the area of the candidate shape. We expect that the puzzle corners will form the maximum area
    - rectangularness: it is the mse of the candidate shape's angles compared to a 90 degree angles. The smaller
                        this value, the most the shape is similar toa rectangle.
        """


    N = len(xy)

    distances = scipy.spatial.distance.cdist(xy, xy)
    distances[distances < d_threshold] = 0


    def compute_angles(xy):
        angles = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):

                point_i, point_j = xy[i], xy[j]
                if point_i[0] == point_j[0]:
                    angle = 90
                else:
                    angle = np.arctan2(point_j[1] - point_i[1], point_j[0] - point_i[0]) * 180 / np.pi

                angles[i, j] = angle
                angles[j, i] = angle

        return angles


    angles = compute_angles(xy)
    possible_rectangles = []


    def search_for_possible_rectangle(idx, prev_points=[]):
        curr_point = xy[idx]
        depth = len(prev_points)

        if depth == 0:
            right_points_idx = np.nonzero(np.logical_and(xy[:, 0] > curr_point[0], distances[idx] > 0))[0]

            if verbose >= 2:
                print
                'point', idx, curr_point

            for right_point_idx in right_points_idx:
                search_for_possible_rectangle(right_point_idx, [idx])

            if verbose >= 2:
                print

            return

        last_angle = angles[idx, prev_points[-1]]
        perp_angle = last_angle - 90
        if perp_angle < 0:
            perp_angle += 180

        if depth in (1, 2):

            if verbose >= 2:
                print
                '\t' * depth, 'point', idx, '- last angle', last_angle, '- perp angle', perp_angle

            diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
            all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))

            diff_to_explore = np.nonzero(np.logical_and(all_diffs, distances[idx] > 0))[0]

            if verbose >= 2:
                print
                '\t' * depth, 'diff0:', np.nonzero(diff0)[0], 'diff180:', np.nonzero(diff180)[
                    0], 'diff_to_explore:', diff_to_explore

            for dte_idx in diff_to_explore:
                if dte_idx not in prev_points:  # unlickly to happen but just to be certain
                    next_points = prev_points[::]
                    next_points.append(idx)

                    search_for_possible_rectangle(dte_idx, next_points)

        if depth == 3:
            angle41 = angles[idx, prev_points[0]]

            diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
            dist = distances[idx, prev_points[0]] > 0

            if dist and (diff0 or diff180_0 or diff180_1):
                rect_points = prev_points[::]
                rect_points.append(idx)

                if verbose == 2:
                    print
                    'We have a rectangle:', rect_points

                already_present = False
                for possible_rectangle in possible_rectangles:
                    if set(possible_rectangle) == set(rect_points):
                        already_present = True
                        break

                if not already_present:
                    possible_rectangles.append(rect_points)


    if verbose >= 2:
        print
        'Coords'
        print
        xy
        print
        print
        'Distances'
        print
        distances
        print
        print
        'Angles'
        print
        angles
        print

    for i in range(N):
        search_for_possible_rectangle(i)

    if len(possible_rectangles) == 0:
        return None


    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


    areas = []
    rectangularness = []
    diff_angles = []

    for r in possible_rectangles:
        points = xy[r]
        areas.append(PolyArea(points[:, 0], points[:, 1]))

        mse = 0
        da = []
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]])
            da.append(abs(diff_angle - 90))
            mse += (diff_angle - 90) ** 2

        diff_angles.append(da)
        rectangularness.append(mse)

    areas = np.array(areas)
    rectangularness = np.array(rectangularness)

    scores = areas * scipy.stats.norm(0, 150).pdf(rectangularness)
    best_fitting_idxs = possible_rectangles[np.argmax(scores)]
    return xy[best_fitting_idxs]

