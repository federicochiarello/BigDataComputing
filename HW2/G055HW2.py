import sys
import os
import time
import numpy as np


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result


def DistanceMatrix(x,y):
    diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix


def SeqWeightedOutliers(P,W,k,z,alpha):

    # Precompute distances
    distance_matrix = DistanceMatrix(np.asarray(P), np.asarray(P))

    # Compute intial value of r
    r_matrix = distance_matrix[:k+z+1, :k+z+1]
    r = min(r_matrix[np.nonzero(r_matrix)]) /2
    initial_r = r
    n_guesses = 1

    while(True):

        Z = np.ones(len(P), dtype=bool)
        S = []
        W_z = np.sum(W)

        while(len(S) < k and W_z > 0):

            max = 0
            newcenter = 0

            for x in range(len(P)):
                              
                # B_z(x,(1+2*alpha)*r)
                x_row_dist = distance_matrix[x]
                candidate_distances = x_row_dist <= (1+2*alpha)*r
                B_z_center = np.logical_and(candidate_distances, Z)

                ball_weight = np.sum(W[B_z_center])

                if ball_weight > max:
                    max = ball_weight
                    newcenter = x

            S.append(P[newcenter])

            # B_z(newcenter,(3+4*alpha)*r)
            newcenter_row_dist = distance_matrix[newcenter]
            candidate_distances = newcenter_row_dist <= (3+4*alpha)*r
            B_z_outlier = np.logical_and(candidate_distances, Z)

            # Remove elements of B_z_outlier from Z and subtract their weights from W_z
            Z = np.logical_xor(B_z_outlier, Z)
            W_z -= np.sum(W[B_z_outlier])
        
        if W_z <= z:
            print(f'Initial guess = {initial_r}\nFinal guess = {r}\nNumber of guesses = {n_guesses}')
            return S
        else:
            r = 2*r
            n_guesses += 1


def ComputeObjective(P,S,z):
    distance_matrix = DistanceMatrix(np.asarray(P), np.asarray(S))
    min_dist = distance_matrix.min(1).tolist()
    for i in range(z):
        min_dist.remove(max(min_dist))
    return max(min_dist)


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
