__author__ = 'Florian Laible'

import numpy as np
import numpy.linalg as nplin
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sb
import os
import re

sb.set_style("ticks")
sb.set_context("talk", font_scale=1.4)

alpha = 14
beta = 2180
Delta = 1
eta = 0.92

x = np.linspace(0, 9, 25)
y = np.linspace(0, 9, 25)

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

x0 = np.array([4, 4, 4, 4, 4, 5, 5, 5, 6])
y0 = np.array([3, 4, 5, 6, 7, 4, 5, 6, 5])

PS = np.zeros((len(x0),len(y0)))
zges = np.zeros((len(xv),len(yv)))
z = np.zeros((len(x0),len(y0)))

def rechenknecht(xi,yi,xj,yj,Delta,alpha,beta,eta):
    z1 = 1/(np.pi*(1+eta))
    z2 = np.pi/(2*Delta**2)
    z3 = sp.erf((xi-xj+(Delta/2))/alpha)-sp.erf((xi-xj-(Delta/2))/alpha)
    z4 = sp.erf((yi-yj+(Delta/2))/alpha)-sp.erf((yi-yj-(Delta/2))/alpha)
    z5 = eta/(beta**2)
    z6 = np.exp(-(((xi-xj)**2)+((yi-yj)**2))/(beta**2))

    z = z1*((z2*(z3*z4))+(z5*z6))
    return(z)

def rechenknecht2(x,y,xi,yi,Delta,alpha,beta,eta,D):
    z1 = 1/(np.pi*(1+eta))
    z2 = np.pi/(2*Delta**2)
    z3 = sp.erf((x-xi+(Delta/2))/alpha)-sp.erf((x-xi-(Delta/2))/alpha)
    z4 = sp.erf((y-yi+(Delta/2))/alpha)-sp.erf((y-yi-(Delta/2))/alpha)
    z5 = eta/(beta**2)
    z6 = np.exp(-(((x-xi)**2)+((y-yi)**2))/(beta**2))

    z = z1*((z2*(z3*z4))+(z5*z6))
    z = D * z
    return(z)

for k in range(len(x0)):
    for l in range(len(x0)):
        PS[k, l] = rechenknecht(x0[k], y0[k], x0[l], y0[l],Delta,alpha,beta,eta)

PSinv = nplin.inv(PS)

Test = PS * PSinv
Test = np.round(Test)

EinerVek = np.ones(len(x0))

Erg = nplin.solve(PS,EinerVek)

for i in range(len(x0)):
    z = rechenknecht2(x0[i],y0[i],xv,yv,Delta,alpha,beta,eta,Erg[i])
    zges = zges + z


#print(Erg)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xv, yv, zges, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()
