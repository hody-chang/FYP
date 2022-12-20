import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import manifold

p = np.random.uniform(-1, 1, 1000)
p[p < 0] = -1
p[p > 0] = 1

x = np.random.uniform(-1, 1, 1000)
y = np.random.uniform(-1, 1, 1000)
'''
y[x == 1] = 0
y[x == -1] = 0
x[y == 1] = 0
x[y == -1] = 0
'''
s = x**2+y**2
for i in range(100):
    if s[i] >= 1:
        x[i] = x[i] / np.sqrt(s[i])
        y[i] = y[i] / np.sqrt(s[i])
z = np.sqrt(1-x**2-y**2)*p
np.isnan(z)

'''
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(141, projection="3d")

ax.scatter(x, y, z)

data = np.array([x, y, z]).T

print(data)

trans_data = (
    manifold.Isomap(n_neighbors=3, n_components=2)
    .fit_transform(data)
    .T
)

ax = fig.add_subplot(142)

plt.scatter(trans_data[0], trans_data[1])


plt.show()
'''