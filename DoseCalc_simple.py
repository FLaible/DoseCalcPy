__author__ = 'Florian Laible'
#!/usr/bin/python
import numpy as np
import scipy as sci
import numpy.linalg as nplin
import scipy.special as sps

import matplotlib
matplotlib.use('Qt5Agg')
import math

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import matplotlib.pyplot as plt
import seaborn as sb
import QBT_creator
import time

start = time.time()


#Eingabe Beginn:
file_dir_data = 'C:/Users/Florian Laible/Desktop/'
file_dir_pics = 'C:/Users/Florian Laible/Desktop/Output_DoseCalc_Pics/'

alpha = 14
beta = 2180
Delta = 10
eta = 0.92

#BT/QBT creator:
Basis = 155
Hoehe = int(np.ceil(np.sqrt(3)*Basis/2))
AbstandX = 5
AbstandY = 5
gap = 25
Quad_BT_y_n = 0
#Eingape Ende

if Quad_BT_y_n == 1:
    BT = QBT_creator.QBT_c(Basis,Hoehe,AbstandX,AbstandY,gap)
    ECP_name = 'QBT_' + str(Basis) + '_' + str(Hoehe) + '_' +str(gap) + '_' + str(AbstandX) + '_' + str(AbstandY)
else:
    BT = QBT_creator.BT_c(Basis,Hoehe,AbstandX,AbstandY,gap)
    ECP_name = 'BT_' + str(Basis) + '_' + str(Hoehe) + '_' +str(gap) + '_' + str(AbstandX) + '_' + str(AbstandY)

pic_dir = file_dir_pics + ECP_name + '.png'

ASum = sum(sum(BT))

A = np.array(BT)
x0 = np.zeros((ASum,1))
y0 = np.zeros((ASum,1))
Erg = np.zeros((ASum,1))

x = np.linspace(-1*A.shape[0], AbstandX*A.shape[0], 100)
y = np.linspace(-1*A.shape[1], AbstandY*A.shape[1], 100)

xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
print('Grid erstellt')
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
    PS = np.zeros((len(x0),len(y0)))
    for k in range(len(x0)):
        print((k/len(x0)*100))
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
    # print(Erg)

    Entries2Delete = []
    for m in range(len(x0)):
        if Erg[m] < 0:
            Entries2Delete.append(m)
            BadCount += 1

    x0neu = np.delete(x0,Entries2Delete)
    y0neu = np.delete(y0,Entries2Delete)
    x0 = x0neu
    y0 = y0neu

for i in range(len(x0)):
    z = rechenknecht2(x0[i],y0[i],xv,yv,Delta,alpha,beta,eta,Erg[i])
    zges = zges + z

MinErg = np.min(Erg)
Erg = Erg / MinErg

Outputfile = open(file_dir_data + 'Output_DoseCalc_simple/' + ECP_name + '.txt','w')
Outputfile.write('D ' + ECP_name + '\n')
Outputfile.write('C 10000' + '\n')
for k in range(len(x0)):
    Outputfile.write('RDOT ' + str(int(x0[k])) + ',' + str(int(y0[k])) + ',' + str(math.ceil(Erg[k])) + '\n')
Outputfile.write('END')


# Plot in Circles
fig = plt.figure()
ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
ax.axis('equal')
ax.set_xlabel('x [pix]')
ax.set_ylabel('y [pix]')
area = np.pi * (20 * Erg/np.max(Erg))**2

plt.scatter(x0, y0, s=area, alpha=0.5)

plt.savefig(file_dir_pics  + ECP_name + '_scatter.png')


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(30, 15)

surf = ax.plot_surface(xv, yv, zges, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x [pix]')
ax.set_ylabel('y [pix]')
ax.set_zlabel('norm. I [a.u.]')
plt.savefig(file_dir_pics  + ECP_name + '_surf.png')


end = time.time()
print(end - start)