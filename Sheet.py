from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib as mpl
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

n_neighbors = 10
n_samples = 1000

random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * (np.pi)

indices = (t < (np.pi - (np.pi / 8))) & (t > (np.pi / 8))
colors = p[indices]

x, y, z = (
    np.sin(t[indices]) * np.cos(p[indices]),
    np.sin(t[indices]) * np.sin(p[indices]),
    np.cos(t[indices]),
)

fig = plt.figure(figsize=(15, 8))
plt.suptitle(
    "Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14
)

ax = fig.add_subplot(151, projection="3d")
ax.scatter(x, y, z, c=colors, cmap=mpl.cm.cool)
ax.set_xlabel("x")
ax.view_init(40, -10)

sphere_data = np.array([x, y, z]).T


##########################Isomap Manifold learning###################################

t0 = time()
trans_data = (
    manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
    .fit_transform(sphere_data)
)
t1 = time()
print("%s: %.2g sec" % ("ISO", t1 - t0))

ax = fig.add_subplot(152)
plt.scatter(trans_data[:, 0], trans_data[:, 1], c=colors, cmap=mpl.cm.cool)
plt.title("%s (%.2g sec)" % ("Isomap", t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis("tight")

ref_point = np.random.rand(1, 2)
a = np.append(trans_data, ref_point, axis=0)

nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(a)
distances, indices = nbrs.kneighbors(a)


points = np.concatenate([sphere_data[indices[-1, 1:8]]])

d = distance.cdist(points, points, 'euclidean')
D = np.exp(d)
w = np.matmul(np.linalg.inv(D), points)
ref = np.exp(np.delete(distances[-1], 0))
ref_x = np.matmul(ref, w[:,0])
ref_y = np.matmul(ref, w[:,1])
ref_z = np.matmul(ref, w[:,2])

s = np.sqrt(ref_x**2+ref_y**2+ref_z**2)
sphere_data = np.append(sphere_data, [[ref_x/s, ref_y/s, ref_z/s]], axis=0)
colors[indices[-1, 1:7]] = 10


colors = np.append(colors, 15)

ax = fig.add_subplot(153, projection="3d")
ax.scatter(sphere_data.T[0], sphere_data.T[1], sphere_data.T[2], c=colors, cmap=mpl.cm.cool)
ax.set_xlabel("x")
ax.view_init(40, -10)

trans_data = (
    manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
    .fit_transform(sphere_data)
)

print("%s: %.2g sec" % ("ISO", t1 - t0))

ax = fig.add_subplot(154)
plt.scatter(trans_data[:, 0], trans_data[:, 1], c=colors, cmap=mpl.cm.cool)
plt.title("%s (%.2g sec)" % ("Isomap", t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis("tight")


print(distance.cdist([trans_data[-1]], ref_point, 'euclidean'))
plt.show()