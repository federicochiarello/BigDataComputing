import sys
import os
import time
import math
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
    
    
# def euclidean(point1,point2):
#     res = 0
#     for i in range(len(point1)):
#         diff = (point1[i]-point2[i])
#         res +=  diff*diff
#     return math.sqrt(res)


def euclidean_distance_matrix(x,y):
    diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix


def SeqWeightedOutliers(P,W,k,z,alpha=0):

    # Precomputed distances
    P_array = np.asarray(P)
    dist_mat = euclidean_distance_matrix(P_array, P_array)

    # Compute r_min
    r_min_mat = dist_mat[:k+z+1, :k+z+1]
    r = min(r_min_mat[np.nonzero(r_min_mat)]) /2

    initial_r = r
    n_guesses = 1

    n_points = len(P)
    
    while(True):

        Z = np.ones(n_points, dtype=bool)
        S = []
        W_z = sum(W)

        while(len(S) < k and W_z > 0):
            max = 0
            newcenter = ()

            for x in range(n_points):
                # row x from distance matrix (distances between x and all points in P)
                x_row_dist = dist_mat[x]
                
                # distances <= (1+2*alpha)*r
                # including distances with points that might not be in Z
                center = x_row_dist <= (1+2*alpha)*r

                B_z_center = np.logical_and(center, Z)

                ball_weight = np.sum(W[B_z_center])

                if ball_weight > max:
                    max = ball_weight
                    newcenter = x

            S.append(P[newcenter])

            newcenter_row_dist = dist_mat[newcenter]
            center = newcenter_row_dist <= (3+4*alpha)*r
            B_z_outlier = np.logical_and(center, Z)

            Z = np.logical_xor(B_z_outlier, Z)
            W_z -= np.sum(W[B_z_outlier])
        
        if W_z <= z:
            print(f'Initial guess = {initial_r}\nFinal guess = {r}\nNumber of guesses = {n_guesses}')

            # Plot Points
            plot_cluster(P,S,P_array[Z],k,z,r)

            return S
        else:
            r = 2*r
            n_guesses += 1


def ComputeObjective(P,S,z):
    P_array = np.asarray(P)
    S_array = np.asarray(S)
    dist_mat = euclidean_distance_matrix(P_array, S_array)
    min_dist = dist_mat.min(1).tolist()
    for x in range(z):
        min_dist.remove(max(min_dist))
    return max(min_dist)


####################################################################
#                           TMP
def plot_cluster(data,solution,outliers,k,z,r):
    df = pd.DataFrame(data)
    s = pd.DataFrame(solution)        

    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1])
    ax.scatter(s[0], s[1], color='red')
    if outliers != []:
        out = pd.DataFrame(outliers)
        ax.scatter(out[0], out[1], color='green')
    for i in range(len(solution)):
        cir = plt.Circle(solution[i], radius=r*3, color='r',fill=False)
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(cir)
    plt.title(f'k={k}   -   z={z}  -  r={r}')
    plt.show()
####################################################################


def main():

    assert len(sys.argv) == 4, "Usage: python G055HW2.py <file_name> <k> <z>"

    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    inputPoints = readVectorsSeq(data_path)

    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    z = sys.argv[3]
    assert z.isdigit(), "z must be an integer"
    z = int(z)

    print(f'Input size n = {len(inputPoints)}\nNumber of centers k = {k}\nNumber of outliers z = {z}')

    # unit weights list
    weights = np.ones(len(inputPoints))

    start = time.time()
    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    executionTime = (time.time() - start) * 1000.0

    objective = ComputeObjective(inputPoints,solution,z)
    
    print(f'Objective function = {objective}\nTime of SeqWeightedOutliers = {executionTime}')

    
if __name__ == "__main__":
    main()
