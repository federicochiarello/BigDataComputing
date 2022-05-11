import sys
import os
import time
import math
from scipy.spatial import distance_matrix
import numpy as np

####################################################################
#                            TMP
import pandas as pd
import matplotlib.pyplot as plt
####################################################################



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


# def dist_matrix(P):
#     P_array = np.asarray(P)
#     dist_mat = distance_matrix(P_array, P_array)
#     print(dist_mat)


def SeqWeightedOutliers(P,W,k,z,alpha=0):

    # Compute r_min
    P_array = np.asarray(P[:k+z+1])
    dist_mat = distance_matrix(P_array, P_array)
    r = min(dist_mat[np.nonzero(dist_mat)]) /2

    initial_r = r
    n_guesses = 1
    
    while(True):

        # P.copy() needed to avoid aliasing
        Z = P.copy()
        S = []
        W_z = sum(W)

        while(len(S) < k and W_z > 0):
            max = 0
            newcenter = ()

            # DUBBIO: x in P o Z
            # stesso risultato, Z leggermente più veloce
            for x in P:
                # lista di indici
                B_z_center = []

                # B_z(x,(1+2*alpha)*r)
                for y in range(len(Z)):
                    if(euclidean(x,Z[y]) <= (1+2*alpha)*r):
                        B_z_center.append(y)

                ball_weight = 0
                for i in B_z_center:
                    ball_weight += W[i]
                if ball_weight > max:
                    max = ball_weight
                    newcenter = x
                # print(f'iteration {x}:\t', B_z_center, '\t', ball_weight, '\t', newcenter)

            S.append(newcenter)
            # print('\nS:\t\t',S)

            # B_z(newcenter,(3+4*alpha)*r)
            B_z_outlier = []
            for y in range(len(Z)):
                if(euclidean(newcenter,Z[y]) <= (3+4*alpha)*r):
                    B_z_outlier.append(y)

            # print(f'B_z out:\t', B_z_outlier)
            # print(f'\nZ:\t\t',Z,'\nW_z:\t\t', W_z)

            for y in range(len(B_z_outlier)-1,-1,-1):
                Z.pop(B_z_outlier[y])
                W_z -= W[B_z_outlier[y]]

            # print(f'\nZ:\t\t',Z,'\nW_z:\t\t', W_z)    
        # print('\nUSCITI DA WHILE INTERNO\n')
        
        if W_z <= z:
            print(f'Initial guess = {initial_r}\nFinal guess = {r}\nNumber of guesses = {n_guesses}')



            plot_cluster(P,S,Z,k,z,r)



            return S
        else:
            r = 2*r
            n_guesses += 1


def ComputeObjective(P,S,z):
    return 0
    # P_array = np.asarray(P)
    # S_array =  np.asarray(S)
    # dist_mat = distance_matrix(P_array, S_array)
    # min_dist = dist_mat.min(1).tolist()
    # print(min_dist)
    # print(dist_mat)
    # for x in range(z):
    #     min_dist.remove(max(min_dist))
    # print(max(min_dist))


####################################################################
#                           TMP
def plot_cluster(data,solution,outliers,k,z,r):
    df = pd.DataFrame(data)
    s = pd.DataFrame(solution)
    out = pd.DataFrame(outliers)

    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1])
    ax.scatter(s[0], s[1], color='red')
    ax.scatter(out[0], out[1], color='green')
    for i in range(len(solution)):
        cir = plt.Circle(solution[i], radius=r*3, color='r',fill=False)
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(cir)
    plt.title(f'k={k}   -   z={z}  -  r={r}')
    plt.show()
####################################################################


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

    print(f'Input size n = {n}\nNumber of centers k = {k}\nNumber of outliers z = {z}')

    # unit weigtht list
    weights = [1] * n

    start = time.time()
    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    executionTime = (time.time() - start) #* 1000.0

    objective = ComputeObjective(inputPoints,solution,z)
    
    print(f'Objective function = {objective}\nTime of SeqWeightedOutliers = {executionTime}')
    print('\nFINE\n')
    print('solution:\t', solution)

    


if __name__ == "__main__":
    main()
