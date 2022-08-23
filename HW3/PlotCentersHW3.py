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
    print("Time to compute objective function: ", str((end-start)*1000), " ms")


    ####################################################################################
    start = time.time()
    inputPoints = inputPoints.collect()
    objective = computeObjective_local(inputPoints, solution, z)
    end = time.time()
    print("\nObjective function LOCAL = ", objective)
    print("Time to compute objective function LOCAL: ", str((end-start)*1000), " ms")
    ####################################################################################



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&



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
# Method plot_center
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
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



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------

    start_r1 = time.time()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1)).cache()
    n_coreset = coreset.count()
    end_r1 = time.time()

    # END OF ROUND 1
    
    #------------- ROUND 2 ---------------------------
    
    start_r2 = time.time()

    elems = coreset.collect()
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
def SeqWeightedOutliers (points, weights, k, z, alpha):

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
            
            P_array = np.asarray(points)
            plot_cluster(points,S,P_array[Z],k,z,r)
            
            return S
        else:
            r = 2*r
            n_guesses += 1



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective_local(points, centers, z):
    distance_matrix = DistanceMatrix(np.asarray(points), np.asarray(centers))
    min_dist = distance_matrix.min(1).tolist()
    for i in range(z):
        min_dist.remove(max(min_dist))
    return max(min_dist)



def computeObjective(points, centers, z):
    coreset = points.mapPartitions(lambda iterator: ComputeObjectivePartition(iterator, centers, z))
    elems = coreset.collect()
    elems.sort()
    return elems[-(z+1)]



def ComputeObjectivePartition(iterator, centers, z):
    # points inside the actual partition
    points = list(iterator)

    # distance matrix between the points in the partition and the centers
    distance_matrix = DistanceMatrix(np.asarray(points), np.asarray(centers))

    # min_dist = array with the minimum distance point-center for every point in the partition
    min_dist = distance_matrix.min(1).tolist()
    sorted_min_dist = np.sort(min_dist)

    # return the biggest z+1 values in sorted_min_dist as a list
    return list(sorted_min_dist[-(z+1):])



# Just start the main program
if __name__ == "__main__":
    main()

