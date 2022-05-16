import numpy as np
# from scipy.spatial import distance_matrix

def DistanceMatrix(x,y):
    diff = np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)
    print(diff[0,1])
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix


points = [(1.0, 1.0, 4.3, 7.1), (1.1, 4.1, 1.0, 4.3), (0.9, 2.9, 4.1, 1.0)]
points_2 = [(1.0, 1.0, 4.3, 7.1), (1.1, 4.1, 1.0, 4.3), (0.9, 2.9, 4.1, 1.0), (0.9, 2.9, 4.1, 1.0)]
p = np.asarray(points)
p_2 = np.asarray(points_2)
d_new = DistanceMatrix(p,p_2)

# d_old = distance_matrix(p,p)

# print(p)
print('\n')
print(d_new)