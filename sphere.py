import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import manifold
from sklearn.utils import check_random_state
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

n_neighbors = 5
n_samples = 1000

random_state = check_random_state(0)
p1 = random_state.rand(n_samples) * (2 * np.pi)
p2 = random_state.rand(n_samples) * (2 * np.pi)
p3 = random_state.rand(n_samples) * (2 * np.pi)

x, y, z, w = (

    np.cos(p1),
    np.sin(p1) * np.cos(p2),
    np.sin(p1) * np.sin(p2) * np.cos(p3),
    np.sin(p1) * np.sin(p2) * np.sin(p3)

)
sphere_data = np.array([x, y, z, w]).T

colors = p1
fig = plt.figure(figsize=(15, 8))
reducer = umap.UMAP()
embedding = reducer.fit_transform(sphere_data)
ax = fig.add_subplot()
ax.scatter(embedding[:, 0], embedding[:, 1], c = p1)
ax.axis("tight")

ref_point = np.random.rand(1, 2)
a = np.append(embedding, ref_point, axis=0)

nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(a)
distances, indices = nbrs.kneighbors(a)

points = np.concatenate([sphere_data[indices[-1, 1:8]]])

d = distance.cdist(points, points, 'euclidean')
D = np.exp(d)
w = np.matmul(np.linalg.inv(D), points)
ref = np.exp(np.delete(distances[-1], 0))
ref_x = np.matmul(ref, w[:, 0])
ref_y = np.matmul(ref, w[:, 1])
ref_z = np.matmul(ref, w[:, 2])
ref_w = np.matmul(ref, w[:, 3])

s = np.sqrt(ref_x ** 2 + ref_y ** 2 + ref_z ** 2 + ref_w ** 2)
sphere_data = np.append(sphere_data, [[ref_x / s, ref_y / s, ref_z / s, ref_w / s]], axis=0)
colors[indices[-1, 1:7]] = 10

embedding = reducer.fit_transform(sphere_data)
print(distance.cdist([embedding[-1]], ref_point, 'euclidean'))

plt.show()
