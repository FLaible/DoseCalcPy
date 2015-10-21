__author__ = 'sei'

__author__ = 'Florian Laible'

import numpy as np
from numba import float64
from numba.decorators import jit

import scipy as sp
import numpy.linalg as nplin

#import matplotlib
#matplotlib.use('Qt5Agg')

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import seaborn as sb


import math

sb.set_style("ticks")
sb.set_context("talk", font_scale=1.4)

alpha = 15#38# nm
gamma = 32#45# nm
zeta = 250#360# nm
nu = 3.49
xi = 6.42
# alpha = 14
# beta = 2180
# Delta = 10
# eta = 0.92

def get_rect():
    x = np.linspace(0,50,8)
    y = np.linspace(0,100,16)

    xv, yv = np.meshgrid(x, y)
    xv = np.ravel(xv)
    yv = np.ravel(yv)
    return xv,yv

def get_rods(dist):
    x1,y1 = get_rect()
    x2,y2 = get_rect()
    x2 += 50+dist

    x = np.concatenate((x1,x2))+20
    y = np.concatenate((y1,y2))+20

    return x,y

prefixes = ["rods"]
structures = [get_rods]
dists = [20,25,30,35,40,45,50,55,60,65,70]

# #file_dir = 'C:/Users/Florian Laible/Desktop/'
# file_dir = './'
#
# #ECP_name = 'BT'
# ECP_name = 'dimer'
# pic_dir = file_dir + ECP_name + '.png'
# AbstandX = 5.0
# AbstandY = 5.0
#
# data = sp.misc.imread(pic_dir)
# data = np.array(data[:,:,0])
# mask = data < 127
# x = np.arange(data.shape[0])
# y = np.arange(data.shape[1])
# xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
#
# x0 = xv[mask]*AbstandX
# y0 = yv[mask]*AbstandY


# r = np.linspace(-100,100)
# a = 1/(1+nu+xi)
# b = 1/alpha**2 * np.exp(-(r/alpha)**2)
# c = nu/(2*gamma**2) * np.exp(-(r/gamma)**2)
# d = xi/(2*zeta**2) * np.exp(-(r/zeta)**2)
# z = a*(b+c+d)
# plt.plot(r,z)
# plt.show()


@jit(float64(float64,float64,float64,float64),nopython=True)
def calc_prox_nb(x,y,x0,y0):
    r = np.sqrt(np.power(x-x0,2)+np.power(y-y0,2))
    #a = 1/(np.pi*(1+nu+xi))
    b = 1/alpha**2 * np.exp(-(r/alpha)**2)
    c = nu/(2*gamma**2) * np.exp(-(r/gamma))
    d = xi/(2*zeta**2) * np.exp(-(r/zeta))
    #z = a*(b+c+d)
    z = (b+c+d)
    # z1 = 1/(np.pi*(1+eta))
    # r = np.sqrt(np.power(x-x0,2)+np.power(y-y0,2))
    # z2 = (1/(alpha**2)) *np.exp((-(r)/(alpha**2)))
    # z3 = (eta/(beta**2))*np.exp((-(r)/(beta**2)))
    # z = z1*(z2+z3)
    # z1 = 1/(np.pi*(1+eta))
    # z2 = np.pi/(2*Delta**2)
    # z3 = math.erf((x-x0+(Delta/2))/alpha)-math.erf((x-x0-(Delta/2))/alpha)
    # z4 = math.erf((y-y0+(Delta/2))/alpha)-math.erf((y-y0-(Delta/2))/alpha)
    # z5 = eta/(beta**2)
    # z6 = np.exp(-(((x-x0)**2)+((y-y0)**2))/(beta**2))
    # z = z1*((z2*(z3*z4))+(z5*z6))
    return(z)

def calc_prox_np(x,y,x0,y0):
    r = np.sqrt(np.power(x-x0,2)+np.power(y-y0,2))
    #a = 1/(np.pi*(1+nu+xi))
    b = 1/alpha**2 * np.exp(-np.power(r/alpha,2))
    c = nu/(2*gamma**2) * np.exp(-(r/gamma))
    d = xi/(2*zeta**2) * np.exp(-(r/zeta))
    #z = a*(b+c+d)
    z = (b+c+d)
    # z1 = 1/(np.pi*(1+eta))
    # r = np.sqrt(np.power(x-x0,2)+np.power(y-y0,2))
    # z2 = (1/(alpha**2)) *np.exp((-(r)/(alpha**2)))
    # z3 = (eta/(beta**2))*np.exp((-(r)/(beta**2)))
    # z = z1*(z2+z3)
    # z1 = 1/(np.pi*(1+eta))
    # z2 = np.pi/(2*Delta**2)
    # z3 = sp.special.erf((x-x0+(Delta/2))/alpha)-sp.special.erf((x-x0-(Delta/2))/alpha)
    # z4 = sp.special.erf((y-y0+(Delta/2))/alpha)-sp.special.erf((y-y0-(Delta/2))/alpha)
    # z5 = eta/(beta**2)
    # z6 = np.exp(-(((x-x0)**2)+((y-y0)**2))/(beta**2))
    # z = z1*((z2*(z3*z4))+(z5*z6))
    return(z)

@jit(float64[:](float64[:],float64[:],float64[:],float64[:,:]),nopython=True)
def calc_sum_nb(x,y,factors,distances):
    doses = np.zeros((len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if distances[i,j] < 500:
                doses[i] += calc_prox_nb(x[i], y[i], x[j], y[j])*factors[j]
    return doses

#@jit
def iter(x0,y0):
    #logpoints = [100,200,300,400,500,600,700,800,900,1000,2000]
    logpoints = np.arange(100,5000,100,dtype=np.int)
    p = np.ones(len(x0),dtype=np.float64)*1
    n_start = len(x0)
    niter = 2000
    k = 0
    err = np.ones(len(x0),dtype=np.float64)*np.inf
    while True:
        k += 1
        fac = 1+((niter-k)/niter)*5
        distances = np.zeros((len(x0),len(x0)))

        for i in range(len(x0)):
            distances[i,:] = np.sqrt(np.power(x0[i]-x0,2)+np.power(y0-y0,2))

        doses = calc_sum_nb(x0,y0,p,distances)

        prev_err = np.sum(np.abs(err))
        err = np.subtract(doses,2.0)

        for i in range(len(x0)):
            prox = calc_prox_np(x0, y0, x0[i], y0[i])
            dist = distances[i,:] < 100
            p += -err*prox*fac*dist
            p[np.where(p<=0)] = 0

        mask = p > 0
        if (len(mask) > n_start/2):
            p = p[mask]
            err = err[mask]
            x0 = x0[mask]
            y0 = y0[mask]
            prox = prox[mask]
            dist = dist[mask]
        else :
            mask = p <= 0
            p[mask] = 1

        if k in logpoints :
            print(str(k)+ " iterations done, "+str(len(x0))+" points, err = "+str(np.sum(np.abs(err))))

        #print(np.sum(np.abs(err)))
        #if (np.sum(np.abs(err))) < 1.0:
        #    break
        if prev_err < np.sum(np.abs(err)):
            break


    print(np.sum(np.abs(err)))
    return x0,y0,p


Outputfile = open('rods.txt','w')

for l in range(len(structures)):
    for k in range(len(dists)):
        print((l,k))

        Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11500, 11500, 5, 5" + '\n')
        Outputfile.write('I 1' + '\n')
        Outputfile.write('C 16000' + '\n')
        Outputfile.write("FSIZE 20 micrometer" + '\n')
        Outputfile.write("UNIT 1 micrometer" + '\n')

        x0,y0 = structures[l](dists[k])
        name = "pics/"+prefixes[l]+"_"+str(dists[k])+".png"

        n = len(x0)
        print(n)

        x0,y0,res = iter(x0,y0)

        x = np.linspace(np.min(x0)-30,np.max(x0)+30,100)
        y = np.linspace(np.min(y0)-30,np.max(y0)+30,100)

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
        zges = np.zeros((len(xv),len(yv)))

        for i in range(len(x0)):
            z = calc_prox_np(xv,yv,x0[i],y0[i])*res[i]
            zges = zges + z

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xv, yv, zges, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #plt.imshow(zges,cmap=cm.coolwarm)
        #plt.show()
        plt.savefig(name)
        plt.close()
        area = np.pi * (20 * res/np.max(res))**2
        plt.scatter(x0, y0, s=area, alpha=0.5)
        name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_scatter.png"
        plt.savefig(name)
        plt.close()
        print(res)
        x0 = x0/1000
        y0 = y0/1000
        res = np.array(np.round(res),dtype=np.int)
        for j in range(len(x0)):
            if res[j] > 1:
                Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((res[j])) + '\n')
        Outputfile.write('END' + '\n')
        Outputfile.write('\n')
        Outputfile.write('\n')

Outputfile.close()