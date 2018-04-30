import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier


n_neighbors = 15

data = np.loadtxt('colors_1500.csv', unpack=True, skiprows=1,delimiter=',')
u = data[0]
g = data[1]
r = data[2]
i = data[3]
z = data[4]

umg = u-g
gmr = g-r
rmi = r-i
imz = i-z
umr = u-r

colors = umg
#colors = np.append(colors, gmr)
#colors = np.append(colors, rmi)
#colors = np.append(colors, imz)
colors = np.append(colors, umr)
colors = colors.reshape(len(u),2)


X = colors

y = np.loadtxt('galaxyzooS2.csv', unpack=True, skiprows=1)
#y = y.reshape(len(u),1)
knc = KNeighborsClassifier(10)
knc.fit(X,y)
y_pred = knc.predict(X)
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform']: 
    #the value assigned to a query point is computed from a simple majority vote of the nearest neighbors
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Galaxy Classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.savefig('galaxy_neighbor2.png')
plt.show()

#---------------------------------------------
"CMD Plots"

gmag = data[1]
rmag = data[2]
imag = data[3]
zmag = data[4]



#------------------------------------------------------------
# Plot the galaxy colors and magnitudes
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(u - g, gmag, '.k', markersize=2)

ax.set_xlim(1, 4.5)
ax.set_ylim(18.1, 13.5)

ax.set_xlabel(r'$\mathrm{u - g}$')
ax.set_ylabel(r'$\mathrm{g_{Mag}}$')

#plt.savefig('cmd_ug.png')
plt.show()

#---------
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(g - r, rmag, '.k', markersize=2)

ax.set_xlim(1, 4.5)
ax.set_ylim(18.1, 13.5)

ax.set_xlabel(r'$\mathrm{g - r}$')
ax.set_ylabel(r'$\mathrm{r_{Mag}}$')

#plt.savefig('cmd_gr.png')
plt.show()

"""#----------
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(r - i, imag, '.k', markersize=2)

ax.set_xlim(1, 4.5)
ax.set_ylim(18.1, 13.5)

ax.set_xlabel(r'$\mathrm{r - i}$')
ax.set_ylabel(r'$\mathrm{i_{Mag}}$')

plt.show()

#----------
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(i - z, imag, '.k', markersize=2)

ax.set_xlim(1, 4.5)
ax.set_ylim(18.1, 13.5)

ax.set_xlabel(r'$\mathrm{i - z}$')
ax.set_ylabel(r'$\mathrm{z_{Mag}}$')

plt.show()
"""
#----------
fig, ax = plt.subplots(figsize=(5, 3.75))
ax.plot(u - r, imag, '.k', markersize=2)

ax.set_xlim(1, 4.5)
ax.set_ylim(18.1, 13.5)

ax.set_xlabel(r'$\mathrm{u - r}$')
ax.set_ylabel(r'$\mathrm{r_{Mag}}$')

#plt.savefig('cmd_ur.png')
plt.show()