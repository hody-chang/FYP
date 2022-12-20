import numpy as np
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf

from scipy.spatial import distance

n_neighbors = 10
n_samples = 1000

random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * np.pi

# Sever the poles from the sphere.
indices = (t < (np.pi - (np.pi / 8))) & (t > (np.pi / 8))
colors = p[indices]
x, y, z = (
    np.sin(t[indices]) * np.cos(p[indices]),
    np.sin(t[indices]) * np.sin(p[indices]),
    np.cos(t[indices]),
)

sphere_data = np.array([x, y, z]).T

trans_data = (
    manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
    .fit_transform(sphere_data)
)

a = np.append(trans_data, [[0, -1]], axis=0)

nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(a)
distances, indices = nbrs.kneighbors(a)

print(indices[-1, 1:7])
'''
points = np.concatenate(([sphere_data[indices[-1, 1]]], [sphere_data[indices[-1, 2]]], [sphere_data[indices[-1, 3]]], [sphere_data[indices[-1, 4]]],
                         [sphere_data[indices[-1, 5]]], [sphere_data[indices[-1, 6]]], [sphere_data[indices[-1, 7]]]))

d = distance.cdist(points, points, 'euclidean')
D = np.exp(d)
w = np.matmul(np.linalg.inv(D), points)
ref = np.exp(np.delete(distances[-1], 0))
ref_x = np.matmul(ref, w[:,0])
ref_y = np.matmul(ref, w[:,1])
ref_z = np.matmul(ref, w[:,2])
print([ref_x, ref_y ,ref_z])
'''

#RBFInterpolator([points.T[0],points.T[1]], points.T[2])

#D = distance.squareform(distance.pdist(trans_data.T))
