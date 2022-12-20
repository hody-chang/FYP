from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib as mpl

n_neighbors = 5
n_samples = 1000

random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
t = random_state.rand(n_samples) * np.pi

# Sever the poles from the sphere.
indices = (t < (np.pi - (np.pi / 8))) & (t > (np.pi / 8))
colors = p[indices]
colors[587] = 10
colors[247] = 10
colors[687] = 10
colors[84] = 10
x, y, z = (
    np.sin(t[indices]) * np.cos(p[indices]),
    np.sin(t[indices]) * np.sin(p[indices]),
    np.cos(t[indices]),
)

x = np.append(x, -0.5045532062356642)
y = np.append(y, 0.1835642730079468)
z = np.append(z, -0.7846912077227081)
colors = np.append(colors, 15)

# Plot our dataset.
fig = plt.figure(figsize=(15, 8))
plt.suptitle(
    "Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14
)

ax = fig.add_subplot(141, projection="3d")
ax.scatter(x, y, z, c=colors, cmap=mpl.cm.cool)
ax.set_xlabel("x")
ax.view_init(40, -10)

sphere_data = np.array([x, y, z]).T

# Perform Isomap Manifold learning.
t0 = time()
trans_data = (
    manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
    .fit_transform(sphere_data)
    .T
)
t1 = time()
print("%s: %.2g sec" % ("ISO", t1 - t0))

ax = fig.add_subplot(142)
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=mpl.cm.cool)
plt.title("%s (%.2g sec)" % ("Isomap", t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis("tight")

print(trans_data.T[-1])
# PCA
pca = PCA(n_components=3)
pca.fit_transform(sphere_data)

t0 = time()
trans_data = pca.fit_transform(sphere_data).T


t1 = time()
ax = fig.add_subplot(143)
plt.scatter(trans_data[0], trans_data[1], c=colors, cmap=plt.cm.rainbow)
plt.title("%s (%.2g sec)" % ("PCA", t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis("tight")

'''
reducer = umap.UMAP()
embedding = reducer.fit_transform(sphere_data)
embedding.shape
ax = fig.add_subplot(144)
ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=plt.cm.rainbow)
ax.axis("tight")
'''
plt.show()