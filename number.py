import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.utils import check_random_state
import umap.umap_ as umap

cos = np.cos
sin = np.sin
sqrt = np.sqrt
pi = np.pi

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

print(np.squeeze([x, y, z]).T)
#sphere_data = np.squeeze(np.array([x, y, z]).T, axis=0)
