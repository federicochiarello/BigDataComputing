from re import X
import sys
import os
import time
import math
from scipy.spatial import distance_matrix
import numpy as np


def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
    
    
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)


def ComputeObjective(P,S,z):
    pass


def SeqWeightedOutliers(P,W,k,z,alpha=0):

    P_array = np.asarray(P[:k+z+1])
    dist_mat = distance_matrix(P_array, P_array)
    r = min(dist_mat[np.nonzero(dist_mat)]) /2
    print(r)

    while(True):
        Z = P.copy()
        S = []
        W_z = sum(W)
        while(len(S) < k and W_z > 0):
            max = 0
            newcenter = ()
            # dubbio x in P o Z
            for x in P:
                # lista di indici
                B_z_center = []

                # B_z(x,(1+2alpha)r)
                for y in range(len(Z)):
                    if(euclidean(x,Z[y]) <= (1+2*alpha)*r):
                        B_z_center.append(y)

                ball_weight = 0
                for i in B_z_center:
                    ball_weight += W[i]
                if ball_weight > max:
                    max = ball_weight
                    newcenter = x
            S.append(newcenter)

            # B_z(newcenter,(3+4alpha)r)
            B_z_outlier = []
            for y in range(len(Z)):
                if(euclidean(newcenter,Z[y]) <= (3+4*alpha)*r):
                    B_z_outlier.append(y)
            
            # print('B_z_out', B_z_outlier)
            # print('len', len(B_z_outlier))

            print(len(Z))
            print(len(P))

            for y in range(len(B_z_outlier)-1,-1,-1):
                print(y)
                Z.pop(B_z_outlier[y])
                W_z -= W[B_z_outlier[y]]

        
        if W_z <= z:
            return S
        else:
            r = 2*r


def printOutput(n,k,z,initial_guess,final_guess,n_guesses,objective,executionTime):
    print(f"""
Input size n = {n}
Number of centers k = {k}
Number of outliers z = {z}
Initial guess = {initial_guess}
Final guess = {final_guess}
Number of guesses = {n_guesses}
Objective function = {objective}
Time of SeqWeightedOutliers = {executionTime}
""")


def main():

    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 4, "Usage: python G055HW2.py <file_name> <k> <z>"

    # INPUT READING

    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    inputPoints = readVectorsSeq(data_path)

    k = sys.argv[2]
    assert k.isdigit(), "k must be an integer"
    k = int(k)

    z = sys.argv[3]
    assert z.isdigit(), "z must be an integer"
    z = int(z)

    # input size
    n = len(inputPoints)

    # unit weigtht list
    weights = [1] * n

    start = time.time()
    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    end = time.time()

    executionTime = (end - start) * 1000.0

    objective = ComputeObjective(inputPoints,solution,z)

    # printOutput(n,k,z,initial_guess,final_guess,n_guesses,objective,executionTime)

    # printOutput(15,3,1,0.04999999999999999,0.7999999999999998,5,1.562049935181331,416)

    print(solution)


if __name__ == "__main__":
    main()
