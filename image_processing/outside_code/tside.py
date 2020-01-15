import numpy as np
import cv2
import time

#optional
def itamar_mefager(japanika):
    return "hello " + japanika

def dist(cord1,cord2):
    return (cord1[0]-cord2[0])**2+(cord1[1]-cord2[1])**2

def make_curve(cord_array, radius):
    start = time.time()

    xmax=np.max(cord_array[:,0])
    ymax = np.max(cord_array[:, 1])
    cord_matrix=np.zeros((xmax+1,ymax+1))

    for cord in cord_array:
        cord_matrix[cord[0],cord[1]]=1
    cord=cord_array[0]
    cnt=0
    results = np.zeros(cord_array.shape)
    while cnt < len(cord_array):
        cord_matrix[cord[0], cord[1]] = 0
        min = 10000
        mincord = None
        for i in range(cord[0]-radius,cord[0]+radius+1):
            for j in range(cord[1]-radius,cord[1]+radius+1):
                if not(i < 0 or i >= len(cord_matrix) or j < 0 or j >= len(cord_matrix[0])):
                    if cord_matrix[i, j] == 1:
                        diq = np.linalg.norm(cord - np.array([i, j]))
                        if diq<min:
                            min = diq
                            mincord = np.array([i, j])
        results[cnt,:] = cord
        cord = mincord
        cnt += 1
    end = time.time()
    print(end-start)
    return results