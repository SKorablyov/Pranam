import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#X = np.arange(1000.0) * 0.001


def camel(X):
    relu1 = np.abs((X - 0.2))
    relu2 = np.abs((X - 0.5))
    relu3 = np.abs((X - 0.7))
    return relu1 - relu2 + relu3


#plt.plot(camel(X + 0.2))
#plt.plot(camel(X + 0.7))

x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 2, 100)
xv, yv = np.meshgrid(x, y)


plt.plot(camel(x))
plt.plot(camel(x - 0.2))
plt.plot(camel(x) + camel(x-0.2))
#plt.show()

fig = plt.figure()
#fig.set_size_inches(12.8, 12.8)
ax = fig.gca(projection='3d')

zv = np.reshape(camel(np.reshape(xv,[-1])),xv.shape) + np.reshape(camel(np.reshape(yv,[-1]) -0.2),xv.shape)

surf = ax.plot_surface(xv, yv, zv ,rstride=1, cstride=1, cmap=cm.coolwarm, color='c', linewidth=0)
plt.show()