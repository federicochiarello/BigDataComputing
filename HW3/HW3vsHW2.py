# Comparison between the results of clustering in HW2 and HW3
# 
# The program also outputs 3 plots:
#   - centers and coreset points from MR_kCenterOutliers
#   - k-centers with z outliers from sequential algorithm
#   - comparison between sequential alg. centers and MR_kCenterOutliers centers


import findspark
findspark.init()


# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms\n")

    # SEQUENTIAL ALGORITHM

    points = readVectorsSeq(filename)
    weights = np.ones(len(points))

    start = time.time()
    seq_solution = SeqWeightedOutliers(points, weights, k, z, alpha=0, plot=True)
    executionTime = (time.time() - start) * 1000.0

    start = time.time()
    objective = ComputeObjectiveLocal(points,seq_solution,z)
    end = time.time()

    print(f'Time of SeqWeightedOutliers = {executionTime}\nObjective function = {objective}')
    print("Time to compute objective function: ", str((end-start)*1000), " ms\n")

    plot_centers(np.asarray(points), solution, seq_solution, k, z)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method readVectorsSeq
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method plot_center
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def plot_centers(data,centers,seq_centers,k,z):

    # print(f'MR_kCenterOutliers:   {centers}')
    # print(f'Sequential Algorithm: {seq_centers}')

    common_centers = list(set(centers) & set(seq_centers))

    df = pd.DataFrame(data)
    s = pd.DataFrame(centers)
    s_seq = pd.DataFrame(seq_centers)    

    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1], color='dimgray', label='Pointset')
    ax.scatter(s[0], s[1], color='dodgerblue', label='MR_kCenterOutliers centers')
    ax.scatter(s_seq[0], s_seq[1], color='limegreen', label='Sequential Algorithm centers')

    if common_centers != []:
        common_centers = pd.DataFrame(common_centers)
        ax.scatter(common_centers[0], common_centers[1], color='red')

    ax.set_aspect('equal', adjustable='datalim')
    plt.title(f'Centers comparison - k={k}, z={z}')
    plt.legend(loc="upper left")
    plt.show()



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method plot_cluster
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def plot_cluster(data,solution,outliers,k,z,r):
    df = pd.DataFrame(data)
    s = pd.DataFrame(solution)        

    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1], color='dodgerblue', label='Pointset')
    ax.scatter(s[0], s[1], color='red', label='Centers')
    if outliers != []:
        out = pd.DataFrame(outliers)
        ax.scatter(out[0], out[1], color='limegreen', label='Outliers')
    for i in range(len(solution)):
        cir = plt.Circle(solution[i], radius=r*3, color='r',fill=False)
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(cir)
    plt.title(f'k-center with z outliers - k={k}, z={z}, r={r}')
    plt.legend(loc="upper left")
    plt.show(block=False)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method plot_coreset
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def plot_coreset(points,coreset,weights,solution,k,z):

    points = points.collect()
    df = pd.DataFrame(points)
    s = pd.DataFrame(solution)
    c = pd.DataFrame(coreset)      

    fig, ax = plt.subplots()
    ax.scatter(df[0], df[1], color='dimgray', label='Pointset')
    ax.scatter(c[0], c[1], weights, color='dodgerblue', label='Coreset Points')
    ax.scatter(s[0], s[1], color='red', label='Centers')

    ax.set_aspect('equal', adjustable='datalim')    
    plt.title(f'Centers and Coreset points - k={k}, z={z}')
    plt.legend(loc="upper left")
    plt.show(block=False)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------

    start_r1 = time.time()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))

    # END OF ROUND 1
    
    #------------- ROUND 2 ---------------------------

    elems = coreset.collect()
    # print(elems)
    end_r1 = time.time()

    start_r2 = time.time()
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution

    coresetWeights = np.asarray(coresetWeights)
    solution = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)

    end_r2 = time.time()

    print("Time Round 1: ", str((end_r1-start_r1)*1000), " ms")
    print("Time Round 2: ", str((end_r2-start_r2)*1000), " ms")

    plot_coreset(points, coresetPoints, coresetWeights, solution, k, z)

    return solution

   

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method DistanceMatrix: compute distance matrix between numpy array
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def DistanceMatrix(x,y):
    # diff.shape = (|x|, |y|, points_dimensionality)
    # diff[i,j] = array with the differences between the coordinates of point i and the coordinates of point j
    diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers (points, weights, k, z, alpha, plot=False):

    # Precompute distances
    distance_matrix = DistanceMatrix(np.asarray(points), np.asarray(points))

    # Compute intial value of r
    r_matrix = distance_matrix[:k+z+1, :k+z+1]
    r = min(r_matrix[np.nonzero(r_matrix)]) /2
    initial_r = r
    n_guesses = 1

    while(True):

        Z = np.ones(len(points), dtype=bool)
        S = []
        W_z = np.sum(weights)

        while(len(S) < k and W_z > 0):

            max = 0
            newcenter = 0

            for x in range(len(points)):
                              
                # B_z(x,(1+2*alpha)*r)
                x_row_dist = distance_matrix[x]
                candidate_distances = x_row_dist <= (1+2*alpha)*r
                B_z_center = np.logical_and(candidate_distances, Z)

                ball_weight = np.sum(weights[B_z_center])

                if ball_weight > max:
                    max = ball_weight
                    newcenter = x

            S.append(points[newcenter])

            # B_z(newcenter,(3+4*alpha)*r)
            newcenter_row_dist = distance_matrix[newcenter]
            candidate_distances = newcenter_row_dist <= (3+4*alpha)*r
            B_z_outlier = np.logical_and(candidate_distances, Z)

            # Remove elements of B_z_outlier from Z and subtract their weights from W_z
            Z = np.logical_xor(B_z_outlier, Z)
            W_z -= np.sum(weights[B_z_outlier])
        
        if W_z <= z:
            print(f'Initial guess = {initial_r}\nFinal guess = {r}\nNumber of guesses = {n_guesses}')
            if plot:
                P_array = np.asarray(points)
                plot_cluster(points,S,P_array[Z],k,z,r)
            return S
        else:
            r = 2*r
            n_guesses += 1



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points, centers, z):
    min_dist = points.mapPartitions(lambda iterator: computeObjectivePartition(iterator, centers, z))
    elems = min_dist.collect()
    elems.sort()
    return elems[-(z+1)]



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjectivePartition: computes objective function in the partition
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjectivePartition(iterator, centers, z):
    # points inside the actual partition
    points = list(iterator)

    # distance matrix between the points in the partition and the centers
    distance_matrix = DistanceMatrix(np.asarray(points), np.asarray(centers))

    # min_dist = array with the minimum distance point-center for every point in the partition
    min_dist = distance_matrix.min(1).tolist()
    sorted_min_dist = np.sort(min_dist)

    # return the biggest z+1 values in sorted_min_dist as a list
    return list(sorted_min_dist[-(z+1):])



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def ComputeObjectiveLocal(P,S,z):
    distance_matrix = DistanceMatrix(np.asarray(P), np.asarray(S))
    min_dist = distance_matrix.min(1).tolist()
    for i in range(z):
        min_dist.remove(max(min_dist))
    return max(min_dist)



# Just start the main program
if __name__ == "__main__":
    main()

