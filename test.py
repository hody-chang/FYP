import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.utils import check_random_state
import umap.umap_ as umap
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

cos = np.cos
sin = np.sin
sqrt = np.sqrt
pi = np.pi
r = 3
n_neighbors = 2
n_samples = 2500

'''
u, v = np.linspace(0, 2*pi, 100), np.linspace(0, 2*pi, 100)
x = (r+cos(u/2)*sin(v)-sin(u/2)*sin(2*v))*cos(u)
y = (r+cos(u/2)*sin(v)-sin(u/2)*sin(2*v))*sin(u)
z = sin(u/2)*sin(v)+cos(u/2)*sin(2*v)
'''

random_state = check_random_state(0)
p = np.linspace(0, 2 * pi, 2500)

u, v = np.linspace(0, pi, 50), np.linspace(0, 2 * pi, 50)
u, v = np.meshgrid(u, v, sparse=True)
U, V = np.meshgrid(u, v, sparse=False)

x = -2 / 15 * np.cos(u) * (3 * np.cos(v) - 30 * np.sin(u) +
                           90 * (np.cos(u) ** 4) * np.sin(u) - 60 * (np.cos(u) ** 6) * np.sin(u) +
                           5 * np.cos(u) * np.cos(v) * np.sin(u))

y = -1 / 15 * np.sin(u) * (3 * np.cos(v) - 3 * (np.cos(u) ** 2) * np.cos(v) -
                           48 * (np.cos(u) ** 4) * np.cos(v) +
                           48 * (np.cos(u) ** 6) * np.cos(v) - 60 * np.sin(u) +
                           5 * np.cos(u) * np.cos(v) * np.sin(u) -
                           5 * (np.cos(u) ** 3) * np.cos(v) * np.sin(u) -
                           80 * (np.cos(u) ** 5) * np.cos(v) * np.sin(u) +
                           80 * (np.cos(u) ** 7) * np.cos(v) * np.sin(u))

z = 2 / 15 * (3 + 5 * np.cos(u) * np.sin(u)) * np.sin(v)

x = np.reshape(x, (1, 2500))
y = np.reshape(y, (1, 2500))
z = np.reshape(z, (1, 2500))

fig = plt.figure()

ax = fig.add_subplot(121, projection="3d")
ax.scatter(x, y, z, c=p, cmap=plt.cm.rainbow)

'''
R = 1
P = 1
e = 10**(-10000)
u, v = np.linspace(0, 2*pi, 100), np.linspace(0, 2*pi, 100)
x = R*(cos(u/2)*cos(v)-sin(u/2)*sin(2*v))
y = R*(sin(u/2)*cos(v)+cos(u/2)*sin(2*v))
z = P*cos(u)*(1+e*sin(v))
w = P*sin(u)*(1+e*sin(v))

p = np.linspace(0, 2*pi, 100)
'''
sphere_data = np.squeeze([x, y, z]).T
colors = p

fig = plt.figure()
reducer = umap.UMAP()
embedding = reducer.fit_transform(sphere_data)
ax = fig.add_subplot(152)
ax.scatter(embedding[:, 0], embedding[:, 1], c=p, cmap=plt.cm.rainbow)
ax.axis("tight")

ax = fig.add_subplot()
plt.scatter(embedding[:, 0], embedding[:, 1], c=p)

ref_point = [0, 2]
a = np.append(embedding, [ref_point], axis=0)

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
s = np.sqrt(ref_x ** 2 + ref_y ** 2 + ref_z ** 2)
sphere_data = np.append(sphere_data, [[ref_x, ref_y, ref_z]], axis=0)

embedding = reducer.fit_transform(sphere_data)
print(distance.cdist([embedding[-1]], [ref_point], 'euclidean'))

plt.show()
