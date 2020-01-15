import numpy as np

def sort_piece_edges(self, scores, rel_index):
    # print(scores)
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if np.average(scores[i][rel_index]) > np.average(
                    scores[j][rel_index]):
                temp = scores[i]
                scores[i] = scores[j]
                scores[j] = temp
    return scores


def compare_piece_color_edge(self, idx1, other):
    scores = []
    for i in range(4):
        scores.append([i, self.compare_edges_color(idx1, other, i)])
    return self.sort_piece_edges(scores, 1)


def compare_piece_color(self, other):
    scores = []
    for i in range(4):
        results = self.compare_piece_color_edge(i, other)
        for j in range(4):
            scores.append([i, results[j][0], results[j][1]])
    return self.sort_piece_edges(scores, 2)


def compare_edges_color(self, idx1, other, idx2):
    color_vector_1 = self._color_vectors[idx1]
    color_vector_2 = other._color_vectors[idx2]

    results = np.zeros((5, len(color_vector_1)))

    np.roll(color_vector_2, -2)
    for i in range(5):
        results[i] = np.abs(np.subtract(color_vector_1, color_vector_2))
        # print(results[i])
        color_vector_2 = np.roll(color_vector_2, 1)

    return np.minimum(np.minimum(np.minimum(results[0], results[1]),
                                 np.minimum(results[2], results[3])),
                      results[4])

