__author__ = 'sei'

__author__ = 'Florian Laible'

import numpy as np
import scipy.integrate as integrate
from numba import float64
from numba.decorators import jit

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sb

import math

import time

sb.set_style("ticks")
sb.set_context("talk", font_scale=1.4)

#alpha = 15#38# nm
#gamma = 45# nm
#zeta = 860#360# nm
#nu = 3.49
#xi = 6.42

#alpha = 14
#beta = 2180
#Delta = 10
#eta = 0.92

#http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
alpha = 32.9 #nm
beta = 2610 #nm
gamma = 4.1 #nm
eta_1 = 1.66
eta_2s = 0.77

radius = 15
#radius = 30

def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def get_circle(r):
    x = np.zeros(0)
    y = np.zeros(0)

    v = np.array( [r,0] )
    #n = 24
    n = 12
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    #if r > 30:
    v = np.array( [r/2,0] )
    n = 6
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i+2*np.pi/(2*n))).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    #if r > 20:
    x = np.hstack( (x,0) )
    y = np.hstack( (y,0) )

    #xv, yv = np.meshgrid(x, y)
    #xv = np.ravel(xv)
    #yv = np.ravel(yv)
    return x,y

def get_trimer(dist,r):
    x = np.zeros(0)
    y = np.zeros(0)
    v = np.array( [0,0.5*(dist+2*r)/np.sin(np.pi/3)] )
    n = 3
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x1,y1 = get_circle(r)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )

    x += 500
    y += 500

    return x,y

def get_dimer(dist,r):
    x1,y1 = get_circle(r)
    x2,y2 = get_circle(r)
    x1 -= (r+dist/2)
    x2 += (r+dist/2)

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500

    return x,y

def get_hexamer(dist,r):
    x = np.zeros(0)
    y = np.zeros(0)
    v = np.array( [0.5*(dist+2*r)/np.sin(np.pi/6),0] )
    n = 6
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x1,y1 = get_circle(r)

        x = np.hstack( (x,x1+x2) )
        y = np.hstack( (y,y1+y2) )

    x += 500
    y += 500

    return x,y


def get_asymdimer(dist,r):
    x1,y1 = get_circle(r)
    r2 = np.sqrt(2)*r
    x2,y2 = get_circle(r2)
    x1 -= r+dist/2
    x2 += r2+dist/2

    x = np.concatenate((x1,x2))+500
    y = np.concatenate((y1,y2))+500

    return x,y

def get_single(dist,r):
    x1,y1 = get_circle(r)

    x = x1+500
    y = y1+500

    return x,y


def get_triple(dist,r):
    x1,y1 = get_circle(r)
    x2,y2 = get_circle(r)
    x3,y3 = get_circle(r)
    x1 -= 2*r+dist
    x2 += 2*r+dist

    x = np.concatenate((x1,x2,x3))+500
    y = np.concatenate((y1,y2,y3))+500

    return x,y

def get_line(dist,r):
    n = int(r/dist)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] += i*dist

    return x,y


current = 100 * 1e-12 # A
dwell_time = 200 * 1e-9 # s
dose_check_radius = 5 # nm
circular_dose_check = True

#outfilename = 'emre2.txt'
#outfilename = 'asymdimer.txt'
#outfilename = 'single.txt'
outfilename = 'pillars_r'+str(radius)+'nm.txt'
#outfilename = 'test.txt'
#outfilename = 'lines.txt'


# prefixes = ["pillar_dimer","pillar_trimer","pillar_hexamer"]#,"pillar_asymdimer","pillar_triple"]
# structures = [get_dimer,get_trimer,get_hexamer]#,get_asymdimer,get_triple]
# dists = [40]
# for i in range(50):
#     dists.append(dists[i]+1)
# #for i in range(10):
# #    dists.append(dists[i] + 5)

prefixes = ["pillar_dimer"]#,"pillar_trimer","pillar_hexamer"]#,"pillar_asymdimer","pillar_triple"]
structures = [get_dimer]#,get_trimer,get_hexamer]#,get_asymdimer,get_triple]
dists = [40]


#radius = 100
#prefixes = ["line"]
#structures = [get_line]#,get_trimer,get_hexamer,get_asymdimer,get_triple]
#dists = [1,2,5,10]

#dists = [40]
#for i in range(30):
#    dists.append(dists[i]+1)



# prefixes = ["pillar_asymdimer"]
# structures = [get_asymdimer]
# dists = [75,80,85,90,95,100,105]
#
#
# prefixes = ["pillar_single"]
# structures = [get_single]
# dists = [0]

#  prefixes = ["pillar_trimer","pillar_hexamer"]
# structures = [get_trimer,get_hexamer]
# dists = [70,75,80,85,90,95,100]

normalization = 2.41701729505915
#http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
@jit(float64(float64,float64,float64,float64),nopython=True)
def calc_prox(x, y, xi, yi):
    r = math.sqrt( math.pow(x-xi,2) + math.pow(y-yi,2) )
    return  (1/normalization) * (1/(math.pi*(1+eta_1*(1+eta_2s)))) * ( (1/(alpha**2))*math.exp(-(r**2)/(alpha**2)) + eta_1*( (1/(beta**2))*math.exp(-(r**2)/(beta**2))+(eta_2s/(24*(gamma**2)))*math.exp(-np.sqrt((r**2)/(alpha**2)))  )   )
# [return] = C/nm !!!

@jit(float64(float64),nopython=True)
def calc_prox_r(r):
    return  (1/normalization) * (1/(math.pi*(1+eta_1*(1+eta_2s)))) * ( (1/(alpha**2))*math.exp(-(r**2)/(alpha**2)) + eta_1*( (1/(beta**2))*math.exp(-(r**2)/(beta**2))+(eta_2s/(24*(gamma**2)))*math.exp(-np.sqrt((r**2)/(alpha**2)))  )   )
# [return] = C/nm !!!


#@jit(float64(float64,float64,float64,float64),nopython=True)
#def calc_prox(x,y,xi,yi):
#    r = np.sqrt( np.square(x-xi) + np.square(y-yi) )
#    return (1/np.pi*(1+eta))*( (1/alpha**2)*np.exp(-np.square(r)/(alpha**2)) + eta/(beta**2)*np.exp(-np.square(r)/(beta^2))      )

#@jit(float64(float64),nopython=True)
#def calc_prox_r(r):
#    return (1/np.pi*(1+eta))*( (1/alpha**2)*np.exp(-np.square(r)/(alpha**2)) + eta/(beta**2)*np.exp(-np.square(r)/(beta^2))      )

#normalization = 34.27477585066464
#@jit(float64(float64,float64,float64,float64),nopython=True)
#def calc_prox(x,y,x0,y0):
#    r = np.sqrt(math.pow(x-x0,2)+math.pow(y-y0,2))
#    b = 1/alpha**2 * math.exp(-(r/alpha)**2)
#    c = nu/(2*gamma**2) * math.exp(-(r/gamma))
#    d = xi/(2*zeta**2) * math.exp(-(r/zeta))
#    z = (1/normalization) * (b+c+d)
#    return(z)

#@jit(float64(float64),nopython=True)
#def calc_prox_r(r):
#    b = 1/alpha**2 * math.exp(-(r/alpha)**2)
#    c = nu/(2*gamma**2) * math.exp(-(r/gamma))
#    d = xi/(2*zeta**2) * math.exp(-(r/zeta))
#    z = (1/normalization)*(b+c+d)
#    return(z)

#normalization, error = integrate.quad(lambda x: calc_prox_r(x)*2*np.pi*x, 0, np.inf)
#print(normalization)
print(integrate.quad(lambda x: calc_prox_r(x)*2*np.pi*x, 0, np.inf))


@jit(float64[:,:](float64[:],float64[:]),nopython=True)
def distance(x, y):
    distances = np.zeros((len(x), len(x)),dtype=np.float64)
    for i in range(len(x)):
        for j in range(len(x)):
            distances[i,j] = math.sqrt(math.pow(x[i] - x[j], 2) + math.pow(y[i] - y[j], 2))
    return distances

@jit(float64[:](float64[:],float64[:,:],float64[:,:]),nopython=True)
def calc_sum(doses,distances, proximity):
    exposure = np.zeros((len(doses)),dtype=np.float64)
    for i in range(len(doses)):
        for j in range(len(doses)):
            if distances[i,j] < 2000:
                exposure[i] += proximity[i,j]*doses[j] #calc_prox_nb(x[i], y[i], x[j], y[j])*factors[j]
    return exposure

@jit(float64[:](float64[:],float64,float64[:,:],float64[:,:],float64[:]),nopython=True)
def calc_correction(err, fac, distances, proximity, repetitions):
    correction = np.zeros((len(err)),dtype=np.float64)
    for i in range(len(repetitions)):
        for j in range(len(repetitions)):
            if not i == j:
                #prox = calc_prox_r_nb(distances[i,j])
                correction[i] += -err[i]*proximity[i,j]*fac#/distances[i,j]
                #repetitions[i] += -err[i] * fac * proximity[i,j]
    return correction

@jit(float64[:](float64[:],float64[:],float64[:],float64[:],float64[:]),nopython=True)
def calc_dose_map(x0, y0, doses, gridx, gridy):
    exposure = np.zeros((len(gridx)),dtype=np.float64)
    pixel_area = np.abs(gridx[0] - gridx[1]) * np.abs(gridx[0] - gridx[1])  # nm^2
    for i in range(len(gridx)):
        for k in range(len(x0)):
            z = calc_prox(x0[k], y0[k], gridx[i], gridy[i]) * doses[k] * pixel_area
            exposure[i] = exposure[i] + z
    return exposure

#@jit(float64[:](float64[:],float64[:],float64[:],float64[:],float64[:]),nopython=True)
#def calc_dose_map_prox(doses, proximity, pixel_area):
#    for i in range(proximity.shape[0]):
#        for k in range(proximity.shape[1]):
#            z = proximity[k,i] * doses[k] * pixel_area
#            exposure[i] = exposure[i] + z
#    return exposure

@jit(float64[:](float64[:],float64[:],float64[:],float64[:],float64[:]),nopython=True)
def calc_dose_grid(x0, y0, doses, gridx, gridy):
    dose = np.zeros((len(x0)),dtype=np.float64)

    for i in range(len(x0)):
        x = gridx+x0[i]
        y = gridy+y0[i]
        exposure = calc_dose_map(x0, y0, doses, x, y) # nm, nm, C
        exposure = exposure * 1e6 # uC

        dose[i] = np.sum(exposure)# uC  /exposure_area # uC/cm^2

    return dose


@jit
def iteration(x0, y0):
    niter = 5000
    logpoints = np.arange(100,niter,100,dtype=np.int)
    repetitions = np.ones(len(x0),dtype=np.float64)*1
    proximity = np.zeros((len(x0),len(x0)),dtype=np.float64)
    doses = np.zeros(len(x0),dtype=np.float64)
    n_start = len(x0)
    k = 0
    areadose_err = np.ones(len(x0),dtype=np.float64)*np.inf

    x = np.linspace(-dose_check_radius, +dose_check_radius, 15)
    y = x
    xgrid, ygrid = np.meshgrid(x, y)
    xgrid = xgrid.ravel()
    ygrid = ygrid.ravel()
    if circular_dose_check:
        mask = xgrid * xgrid + ygrid * ygrid <= dose_check_radius * dose_check_radius
        xgrid = xgrid[mask]
        ygrid = ygrid[mask]
    pixel_area = np.abs(xgrid[0] - xgrid[1]) * np.abs(xgrid[0] - xgrid[1])  # nm^2
    pixel_area = pixel_area * 1e-14  # cm^2
    exposure_area = pixel_area * (len(xgrid))  # cm^2

    starttime = time.time()
    while True:
        k += 1
        fac = 30+((niter-k)/niter)*50

        distances = distance(x0, y0)

        for i in range(len(x0)):
            for j in range(len(x0)):
                proximity[i,j] = calc_prox_r(distances[i, j])

        doses = repetitions * current * dwell_time

        areadose = calc_dose_grid(x0, y0, doses, xgrid.ravel(), ygrid.ravel()) # uC
        areadose = areadose/exposure_area # uC/cm^2

        prev_err = np.sum(np.abs(areadose_err))/len(repetitions)
        areadose_err = np.subtract(areadose, 300)

        areadose_correction = calc_correction(areadose_err, fac, distances, proximity, repetitions)

        repetitions += (areadose_correction * exposure_area * 1e-6) / (current * dwell_time)

        repetitions[np.where(repetitions<=0.1)] = 0

        mask = repetitions > 0
        if (len(mask) > n_start/2):
            repetitions = repetitions[mask]
            areadose_err = areadose_err[mask]
            x0 = x0[mask]
            y0 = y0[mask]

        else :
            mask = repetitions <= 0
            repetitions[mask] = 1

        new_err = np.sum(np.abs(areadose_err))/len(repetitions)

        if k in logpoints :
            print(str(k)+ " iterations done, "+str(len(repetitions))+" points, err = "+str(new_err))

        if (prev_err < new_err) and (k > 2000):
            print("early termination due to rising error")
            break

        if new_err < 0.5*1e-1:
            print("terminated iteration, error is sufficiently small")
            break

        if k > niter:
            break

    #print(new_err)
    return x0,y0,repetitions


Outputfile = open(outfilename,'w')

for l in range(len(structures)):
    for k in range(len(dists)):

        Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11500, 11500, 5, 5" + '\n')
        Outputfile.write('I 1' + '\n')
        Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
        Outputfile.write("FSIZE 20 micrometer" + '\n')
        Outputfile.write("UNIT 1 micrometer" + '\n')

        x0,y0 = structures[l](dists[k],radius)
        randind = np.random.permutation(len(x0))
        x0 = x0[randind]
        y0 = y0[randind]

        starttime = time.time()
        x0, y0, repetitions = iteration(x0, y0)
        print("time for iteration: "+ str(np.round(time.time()-starttime,2))+" seconds")

        x = np.linspace(np.min(x0)-50,np.max(x0)+50,500)
        y = x
        x, y = np.meshgrid(x, y)
        orig_shape = x.shape
        x = x.ravel()
        y = y.ravel()
        exposure = calc_dose_map(x0, y0, repetitions * dwell_time * current, x, y) # C
        exposure = exposure.reshape(orig_shape)
        exposure = exposure * 1e6 # uC
        pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
        pixel_area = pixel_area * 1e-14  # cm^2
        exposure = exposure/pixel_area # uC/cm^2

        name = "pics/"+prefixes[l]+"_"+str(dists[k])+".png"
        fig = plt.figure()
        cmap = sb.cubehelix_palette(light=1, as_cmap=True,reverse=False)
        plot = plt.imshow(exposure,cmap=cmap,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
        plt.colorbar()
        plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure, [300])
        plt.savefig(name)
        plt.close()

        name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_expected.png"
        fig = plt.figure()
        plot = plt.imshow(exposure >= 300,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
        plt.savefig(name)
        plt.close()


        area = np.pi * (15*repetitions/np.max(repetitions))**2
        plt.scatter(x0, y0, s=area, alpha=0.5,edgecolors="black",linewidths=1)
        plt.axes().set_aspect('equal', 'datalim')
        name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_scatter.png"
        plt.savefig(name)
        plt.close()
        x0 = x0/1000
        y0 = y0/1000
        repetitions = np.array(np.round(repetitions),dtype=np.int)
        print(repetitions)
        for j in range(len(x0)):
            if repetitions[j] > 1:
                Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
        Outputfile.write('END' + '\n')
        Outputfile.write('\n')
        Outputfile.write('\n')

Outputfile.close()