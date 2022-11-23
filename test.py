import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

N = 600
dim = 4

norm = np.random.normal
normal_deviates = norm(size=(dim, N))

radius = np.sqrt((normal_deviates**2).sum(axis=0))
points = normal_deviates/radius

print(points)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.scatter(*points)
ax.set_aspect('equal')
plt.show()