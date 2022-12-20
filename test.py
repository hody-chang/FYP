import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.utils import check_random_state

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
p = np.linspace(0, 2*pi, 2500)

u, v = np.linspace(0, pi, 50), np.linspace(0, 2*pi, 50)
u, v = np.meshgrid(u, v, sparse=True)
U, V = np.meshgrid(u, v, sparse=False)

x = -2/15*np.cos(u)*(3*np.cos(v) - 30*np.sin(u) +
90*(np.cos(u)**4)*np.sin(u)-60*(np.cos(u)**6)*np.sin(u)+
5*np.cos(u)*np.cos(v)* np.sin(u))

y = -1/15*np.sin(u)*(3*np.cos(v)-3*(np.cos(u)**2)*np.cos(v)-
48*(np.cos(u)**4)*np.cos(v)+
48*(np.cos(u)**6)*np.cos(v)-60*np.sin(u)+
5*np.cos(u)*np.cos(v)*np.sin(u)-
5*(np.cos(u)**3)*np.cos(v)*np.sin(u)-
80*(np.cos(u)**5)*np.cos(v)*np.sin(u)+
80*(np.cos(u)**7)*np.cos(v)*np.sin(u))


z = 2/15*(3+5*np.cos(u)*np.sin(u))*np.sin(v)

x = np.reshape(x, (1, 2500))
y = np.reshape(y, (1, 2500))
z = np.reshape(z, (1, 2500))

fig = plt.figure()

ax = fig.add_subplot(121, projection="3d")
ax.scatter(x, y, z, c=p, cmap=plt.cm.rainbow)


sphere_data = np.array([x, y, z]).T

trans_data = (
    manifold.Isomap(n_neighbors=n_neighbors, n_components=2)
    .fit_transform(sphere_data[:, 0])
)
ax = fig.add_subplot(122)
plt.scatter(trans_data[:, 0], trans_data[:, 1], c=p, cmap=plt.cm.rainbow)


plt.show()