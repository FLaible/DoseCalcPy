__author__ = 'Florian Laible'
#!/usr/bin/python
import numpy as np
import scipy as sci
import numpy.linalg as nplin
import scipy.special as sps
from numba import jit,float64,int64
import matplotlib
#matplotlib.use('Qt5Agg')
import math
import os



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import matplotlib.pyplot as plt
import seaborn as sb
#import QBT_creator
import time

start = time.time()

#Eingabe Beginn:
path = os.getcwd()
file_dir_data = path
file_dir_pics = path
#print(path)
Name = 'g_500_r_50'
g_x_nm = 500
R_0 = 50

#Name = input('Dateiname der txt Output-Datei: ')

#while int(g_x_nm) < 150:
#    g_x_nm = input('Gitterkonstante in nm: ')
#    if int(g_x_nm) < 150:
#        print('Gitterkonstant zu klein (150 nm ist das untere Limit)')

#while int(R_0) < 10:
#    R_0 = input('Basis Repetitionen pro Punkt: ')
#    if int(g_x_nm) < 10:
#        print('Basis Repetitionen (10 ist das untere Limit)')

g_x_nm = int(g_x_nm)
#R_0 = int(R_0)

g_y_nm = g_x_nm

fsize_nm = 25000
pixels = 50000
fp_factor = pixels/fsize_nm

g_x_pix = g_x_nm * fp_factor
g_y_pix = g_y_nm * fp_factor


n_x = int(np.floor(pixels/g_x_pix))
n_y = int(np.floor(pixels/g_y_pix))

n = n_x * n_y
#print(n)
x = np.zeros((n,1))
y = np.zeros((n,1))

m = 0
for i in range(int(np.floor(pixels/g_x_pix))):
    for j in range(int(np.floor(pixels/g_y_pix))):
        x[m] = i*g_x_pix
        y[m] = j*g_y_pix
        m += 1

alpha = 32.9 #nm
beta = 2610 #nm
gamma = 4.1 #nm
eta_1 = 1.66
eta_2 = 1.27

alpha = 32.9 * fp_factor #pix
beta = 2610 * fp_factor #pix
gamma = 4.1 * fp_factor #pix


def rechenknecht(xi,yi,xj,yj,gamma,alpha,beta,eta_1,eta_2):
    r = math.sqrt( (xi-xj)*(xi-xj)+(yi-yj)*(yi-yj) )
    z = 1 / (math.pi * (1 + eta_1 + eta_2))*((1 / (alpha ** 2)) * math.exp(-r ** 2 / alpha ** 2) + (eta_1 / beta ** 2) * math.exp(-r ** 2 / beta ** 2) + (eta_2 / (24 * gamma ** 2)) * math.exp(-math.sqrt(r / gamma)))
    return(z)

z = np.zeros((len(x),1))
PS = np.zeros((len(x),len(y)),float)
for k in range(len(x)):
    Progress = (k/len(x)*100)
    print('Progress : %3.2f' % Progress + '%')
    for l in range(len(y)):
        PS[k, l] = rechenknecht(x[k], y[k], x[l], y[l],gamma,alpha,beta,eta_1,eta_2)
        z[k] = z[k]+PS[k,l]

EinerVek = np.ones(len(x))
Erg = nplin.solve(PS,EinerVek)
Erg = abs(np.round((((1-Erg/np.min(Erg))*100)),2))

Outputfile = open(file_dir_data + '\\' + str(Name) + '.txt','w')
Outputfile.write('D ' + str(Name) + '\n')
Outputfile.write('C 10000' + '\n')
for k in range(len(x)):
    Outputfile.write('RDOT ' + str(int(x[k])) + ',' + str(int(y[k])) + ',' + str(int(np.round(int(R_0) * (1+(Erg[k]))))) + '\n')
Outputfile.write('END')

end = time.time()
print('Calc-time: t = ' + str(end - start)+ 's')
#print(x)
# Plot in Circles
fig = plt.figure()
ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
ax.axis('equal')
ax.set_xlabel('x [pix]')
ax.set_ylabel('y [pix]')
area = np.pi * ( Erg/np.max(Erg))**2

plt.scatter(x, y, s=area, alpha=0.5)
plt.savefig(file_dir_data + '\\' + str(Name) + '_scatter.png')




