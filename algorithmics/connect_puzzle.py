import numpy as np

from constants import *
from main import *


def get_angle_from_trig(cos, sin):
    if (sin == 0 and cos < 0):
        return np.pi
    if cos > 0:
        return np.arcsin(sin)
    elif (sin > 0):
        return -np.pi - np.arcsin(sin)
    else:
        return np.pi - np.arcsin(sin)


def create_trans_goldner(corner1_start, corner1_end, corner2_start, corner2_end):
    delta_x_end = corner1_end[0] - corner2_end[0]
    delta_y_end = corner1_end[1] - corner2_end[1]
    delta_x_start = corner1_start[0] - corner2_start[0]
    delta_y_start = corner1_start[1] - corner2_start[1]
    delta_dist1 = np.sqrt(delta_x_start ** 2 + delta_y_start ** 2)
    delta_dist2 = np.sqrt(delta_x_end ** 2 + delta_y_end ** 2)
    cos = (delta_y_start * delta_y_end + delta_x_start * delta_x_end) / (delta_dist1 * delta_dist2)
    sin = (delta_y_start * delta_x_end - delta_x_start * delta_y_end) / (delta_dist1 * delta_dist2)
    x0 = delta_x_end - cos * corner1_start[0] - sin * corner1_start[1]
    y0 = delta_y_end + sin * corner1_start[0] - cos * corner1_start[1]
    matrix = np.array([[cos, sin, x0],
                       [-sin, cos, y0], ])

    return matrix

def create_trans(center, corner1_start, corner1_end, corner2_start, corner2_end):
    delta_x_end = corner1_end[0] - corner2_end[0]
    delta_y_end = corner1_end[1] - corner2_end[1]
    delta_x_start = corner1_start[0] - corner2_start[0]
    delta_y_start = corner1_start[1] - corner2_start[1]
    delta_dist1 = np.sqrt(delta_x_start ** 2 + delta_y_start ** 2)
    delta_dist2 = np.sqrt(delta_x_end ** 2 + delta_y_end ** 2)
    cos = (delta_y_start * delta_y_end + delta_x_start * delta_x_end) / (delta_dist1 * delta_dist2)
    sin = (delta_y_start * delta_x_end - delta_x_start * delta_y_end) / (delta_dist1 * delta_dist2)

    alpha = -cos
    beta = -sin
    center_x = center[0]
    center_y = center[1]

    rot_mat = np.array([
        [alpha, beta, (1 - alpha) * center_x - beta * center_y],
        [-beta, alpha, beta * center_x + (1 - alpha) * center_y],
        [0, 0, 1]
    ])

    new_corner = do_transform(corner1_start, rot_mat)
    x0 = corner2_end[0] - new_corner[0]
    y0 = corner2_end[1] - new_corner[1]

    trans_mat = np.array([
        [1, 0, x0],
        [0, 1, y0],
        [0, 0, 1]
    ])

    mat = trans_mat.dot(rot_mat)[:2, :]
    return mat


def get_center_pixel(piece, corner1_start, corner1_end, corner2_start, corner2_end):
    (centerx, centery) = piece.get_real_centroid()
    matrix = create_trans(piece.get_real_centroid(), corner1_start, corner1_end, corner2_start, corner2_end)
    newx = (centerx * matrix[0, 0] + centery * matrix[0, 1] + matrix[0, 2])
    newy = centerx * matrix[1, 0] + centery * matrix[1, 1] + matrix[1, 2]
    sin = matrix[1, 0]
    cos = matrix[0, 0]
    angle = get_angle_from_trig(cos, sin)
    return (np.array([newx, newy]), angle)


def get_point_pixel(piece, corner1_start, corner1_end, corner2_start, corner2_end, point):
    (centerx, centery) = point
    matrix = create_trans(piece.get_real_centroid(), corner1_start, corner1_end, corner2_start, corner2_end)
    newx = centerx * matrix[0, 0] + centery * matrix[0, 1] + matrix[0, 2]
    newy = centerx * matrix[1, 0] + centery * matrix[1, 1] + matrix[1, 2]
    return np.array([newx, newy])


def do_transform(point, matrix):
    x, y = point[0], point[1]
    newx = x * matrix[0, 0] + y * matrix[0, 1] + matrix[0, 2]
    newy = x * matrix[1, 0] + y * matrix[1, 1] + matrix[1, 2]
    return np.array([newx, newy])



def transform_piece(piece, corner1_start, corner1_end, corner2_start, corner2_end):
    matrix = create_trans(piece.get_real_centroid(), corner1_start, corner1_end, corner2_start, corner2_end)
    img = piece.display_real_piece()

    small = img.copy()
    # cv2.circle(small, tuple(corner1_start.astype(dtype=np.int).tolist()), 10, (0, 0, 255), -1)
    # cv2.circle(small, tuple(corner2_start.astype(dtype=np.int).tolist()), 10, (0, 0, 255), -1)
    # cv2.imshow("shalom", cv2.resize(small, (0, 0), fx=0.5, fy=0.5))
    # cv2.waitKey(0)

    dst0 = cv2.warpAffine(img[:, :, 0], matrix, (PUZZLE_SIZE, PUZZLE_SIZE))
    dst1 = cv2.warpAffine(img[:, :, 1], matrix, (PUZZLE_SIZE, PUZZLE_SIZE))
    dst2 = cv2.warpAffine(img[:, :, 2], matrix, (PUZZLE_SIZE, PUZZLE_SIZE))
    dst = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE, 3)).astype(np.uint8)
    dst[:, :, 0] = dst0
    dst[:, :, 1] = dst1
    dst[:, :, 2] = dst2

    small = dst.copy()
    # cv2.circle(small, tuple(corner1_end.astype(dtype=np.int).tolist()), 10, (0, 0, 255), -1)
    # cv2.circle(small, tuple(corner2_end.astype(dtype=np.int).tolist()), 10, (0, 0, 255), -1)
    # cv2.imshow("shalom", cv2.resize(small, (0, 0), fx=0.5, fy=0.5))
    # cv2.waitKey(0)

    return dst


START_X = PUZZLE_SIZE - 1

def get_solved_puzzle_img(final_puzzel):
    # compute starting corners
    piece, edge = final_puzzel[0][0]
    corners = piece.get_real_corners()

    print(edge)

    corner1 = corners[(edge + 1) % 4]
    corner2 = corners[(edge + 2) % 4]
    a = np.sqrt((corner1[0] - corner2[0]) ** 2 + (corner1[1] - corner2[1]) ** 2)

    corner1_next = np.array([START_X, 0])
    corner2_next = np.array([START_X, a])

    n = max([len(row) for row in final_puzzel])
    m = (len(final_puzzel))
    centers = [[None for j in range(n)] for i in range(m)]
    big_pic = np.zeros((PUZZLE_SIZE, PUZZLE_SIZE, 3)).astype(dtype=np.uint8)
    connected_images = []

    for i, row in enumerate(final_puzzel):
        for j, pair in enumerate(row):
            piece, edge = pair

            # start of row
            if j == 0:
                corner1_start = piece.get_real_corners()[(edge + 2) % 4]
                corner2_start = piece.get_real_corners()[(edge + 1) % 4]

                corner1_end = corner1_next
                corner2_end = corner2_next

                corner1_next = get_point_pixel(
                    piece, corner1_start, corner1_end, corner2_start, corner2_end,
                    piece.get_real_corners()[edge % 4]
                )
                corner2_next = get_point_pixel(
                    piece, corner1_start, corner1_end, corner2_start, corner2_end,
                    piece.get_real_corners()[(edge + 3) % 4]
                )
            # regular pieces
            else:
                corner1_start = piece.get_real_corners()[(edge) % 4]
                corner2_start = piece.get_real_corners()[(edge + 1) % 4]

            # print("Transforming %s, moving %s to %s, and %s to %s" % (
            #     piece, corner1_start, corner1_end, corner2_start, corner2_end
            # ))
            img = transform_piece(piece, corner1_start, corner1_end, corner2_start, corner2_end)

            # save piece rotation
            (center, angle) = get_center_pixel(piece, corner1_start, corner1_end, corner2_start, corner2_end)
            centers[i][j] = (piece, center, angle)

            big_pic = big_pic + img
            connected_images.append(big_pic.copy())

            # plot the start circles
            # cv2.circle(big_pic, tuple(corner1_start.astype(dtype=np.int).tolist()), 10, (0, 0, 255), -1)
            # cv2.circle(big_pic, tuple(corner2_start.astype(dtype=np.int).tolist()), 10, (0, 0, 255), -1)
            # cv2.circle(big_pic, tuple(corner1_end.astype(dtype=np.int).tolist()), 10, (0, 255, 255), -1)
            # cv2.circle(big_pic, tuple(corner2_end.astype(dtype=np.int).tolist()), 10, (0, 255, 255), -1)

            corner1_end_new = get_point_pixel(
                piece, corner1_start, corner1_end, corner2_start, corner2_end,
                piece.get_real_corners()[(edge + 2) % 4]
            )
            corner2_end_new = get_point_pixel(
                piece, corner1_start, corner1_end, corner2_start, corner2_end,
                piece.get_real_corners()[(edge + 3) % 4]
            )
            corner1_end = corner1_end_new
            corner2_end = corner2_end_new

    return centers, connected_images


