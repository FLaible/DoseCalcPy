__author__ = 'Florian Laible'

import numpy as np
import scipy as sp
import numpy.linalg as nplin
import scipy.special as sps
import scipy.ndimage as sciim

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sb
import os
import re

#import easygui
#filedir = easygui.fileopenbox()
#print(filedir)


sb.set_style("ticks")
sb.set_context("talk", font_scale=1.4)


alpha = 14
beta = 2180
Delta = 1
eta = 0.92

#fig = plt.figure()
#ax1 = fig.add_axes([0.1, 0.12, 0.8, 0.8])

AbstandX = 10
AbstandY = 10

A = sciim.imread('C:/Users/Florian Laible/Desktop/Kreis.png')
A = A[:,:,0]

for i in range(A.size):
            if A.item(i)<190:
                A.itemset(i,1)
            else:
                A.itemset(i,0)

ASum = sum(sum(A))

x0 = np.zeros((ASum,1))
y0 = np.zeros((ASum,1))
Erg = np.zeros((ASum,1))
#print(A)
#ax1.imshow(A)
#plt.show()

x = np.linspace(-1*A.shape[0], AbstandX*A.shape[0], 100)
y = np.linspace(-1*A.shape[1], AbstandY*A.shape[1], 100)

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
m = 0
for k in range(A.shape[0]):
    for l in range(A.shape[1]):
        if A[k,l] == 1:
            x0[m] = k * AbstandX
            y0[m] = l * AbstandY
            m += 1


zges = np.zeros((len(xv),len(yv)))
z = np.zeros((len(x0),len(y0)))

def rechenknecht(xi,yi,xj,yj,Delta,alpha,beta,eta):
    z1 = 1/(np.pi*(1+eta))
    z2 = np.pi/(2*Delta**2)
    z3 = sps.erf((xi-xj+(Delta/2))/alpha)-sps.erf((xi-xj-(Delta/2))/alpha)
    z4 = sps.erf((yi-yj+(Delta/2))/alpha)-sps.erf((yi-yj-(Delta/2))/alpha)
    z5 = eta/(beta**2)
    z6 = np.exp(-(((xi-xj)**2)+((yi-yj)**2))/(beta**2))

    z = z1*((z2*(z3*z4))+(z5*z6))
    return(z)

def rechenknecht2(x,y,xi,yi,Delta,alpha,beta,eta,D):
    z1 = 1/(np.pi*(1+eta))
    z2 = np.pi/(2*Delta**2)
    z3 = sps.erf((x-xi+(Delta/2))/alpha)-sps.erf((x-xi-(Delta/2))/alpha)
    z4 = sps.erf((y-yi+(Delta/2))/alpha)-sps.erf((y-yi-(Delta/2))/alpha)
    z5 = eta/(beta**2)
    z6 = np.exp(-(((x-xi)**2)+((y-yi)**2))/(beta**2))

    z = z1*((z2*(z3*z4))+(z5*z6))
    z = D * z
    return(z)

BadCount = 1

while BadCount != 0:
    BadCount = 0
    print(len(x0))
    PS = np.zeros((len(x0),len(y0)))
    for k in range(len(x0)):
        for l in range(len(y0)):
            PS[k, l] = rechenknecht(x0[k], y0[k], x0[l], y0[l],Delta,alpha,beta,eta)

    PSinv = nplin.inv(PS)
    Test = PS * PSinv
    Test = np.round(Test)
    EinerVek = np.ones(len(x0))
    Erg = nplin.solve(PS,EinerVek)
    #Erg = sp.linalg.solve_triangular(PS,EinerVek)
    #Erg = sp.sparse.linalg.spsolve(PS,EinerVek)
    #Erg = sp.sparse.linalg.minres(PS,EinerVek)
    print(Erg.shape)

    # Entries2Delete = []
    # for m in range(len(x0)):
    #     #if Erg[m] < 0:
    #     if Erg.any(0):
    #         Entries2Delete.append(m)
    #         BadCount += 1
    #
    # x0neu = np.delete(x0,Entries2Delete)
    # y0neu = np.delete(y0,Entries2Delete)
    # x0 = x0neu
    # y0 = y0neu




for i in range(len(x0)):
    z = rechenknecht2(x0[i],y0[i],xv,yv,Delta,alpha,beta,eta,Erg[i])
    zges = zges + z

print(Test)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xv, yv, zges, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()
