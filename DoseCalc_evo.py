import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit,float64,int64

import math

import time

sns.set_style("ticks")
sns.set_context("talk")

alpha = 32.9 #nm
beta = 2610 #nm
gamma = 4.1 #nm
eta_1 = 1.66
eta_2 = 1.27


radius = 15
#radius = 30

def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def get_circle(r,n=12,inner_circle=False):
    x = np.zeros(0)
    y = np.zeros(0)

    v = np.array( [r,0] )
    #n = 24
    #n = 12
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    if inner_circle:
        v = np.array( [r/2,0] )
        n = int(n/2)
        for i in range(n):
            x2, y2 = (v*rot(2*np.pi/n*i+2*np.pi/(2*n))).A1
            x = np.hstack( (x,x2) )
            y = np.hstack( (y,y2) )

    # #if r > 20:
    #x = np.hstack( (x,0) )
    #y = np.hstack( (y,0) )

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
dose_check_radius = 7 # nm


#outfilename = 'emre2.txt'
#outfilename = 'asymdimer.txt'
#outfilename = 'single.txt'
outfilename = 'pillars2_r'+str(radius)+'nm.txt'
#outfilename = 'test.txt'
#outfilename = 'lines.txt'


prefixes = ["pillar_dimer","pillar_trimer","pillar_hexamer"]#,"pillar_asymdimer","pillar_triple"]
structures = [get_dimer,get_trimer,get_hexamer]#,get_asymdimer,get_triple]
dists = [40]
for i in range(50):
    dists.append(dists[i]+1)
#for i in range(10):
#    dists.append(dists[i] + 5)

# prefixes = ["pillar_dimer"]#,"pillar_trimer","pillar_hexamer"]#,"pillar_asymdimer","pillar_triple"]
# structures = [get_dimer]#,get_trimer,get_hexamer]#,get_asymdimer,get_triple]
# dists = [40]



normalization = 2.41701729505915
#http://iopscience.iop.org/article/10.1143/JJAP.35.1929/pdf
@jit(float64(float64),nopython=True)
def calc_prox(r):
    return (1/normalization) * (1/(math.pi*(1+eta_1+eta_2))) * ( (1/(alpha**2))*math.exp(-r**2/alpha**2) + (eta_1/beta**2)*math.exp(-r**2/beta**2) + (eta_2/(24*gamma**2))*math.exp(-math.sqrt(r/gamma)) )
# [return] = C/nm !!!

@jit(float64(float64,float64,float64,float64),nopython=True)
def dist(x0,y0,x,y):
    return math.sqrt( (x0-x)*(x0-x)+(y0-y)*(y0-y) )


@jit(float64[:](float64[:],float64[:],float64[:],float64[:],float64[:]),nopython=True)
def calc_map_2(x0,y0,doses,x,y):
    exposure = np.zeros(len(x),dtype=np.float64)
    pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
    for i in range(len(x)):
        for j in range(len(x0)):
            exposure[i] += calc_prox(dist(x0[j],y0[j],x[i],y[i]))*doses[j]* pixel_area
    return exposure

@jit(float64[:](float64[:,:],float64[:]),nopython=True)
def calc_map(proximity,doses):
    exposure = np.zeros(proximity.shape[1],dtype=np.float64)
    for i in range(proximity.shape[1]):
        for j in range(proximity.shape[0]):
            exposure[i] += proximity[j,i]*doses[j]
    return exposure

@jit(float64[:, :](float64[:], float64[:]),nopython=True)
def recombine_arrays(arr1, arr2):
    res = np.zeros((len(arr1), 2), dtype=np.float64)
    res[:, 0] = arr1
    res[:, 1] = arr2
    n_crossover = int(len(arr1)/3)
    for i in range(n_crossover):
        k = np.random.randint(0, len(arr1) - 1)
        alpha = np.random.random()
        res[k, 0] = alpha * arr1[k] + (1 - alpha) * arr2[k]
        res[k, 1] = alpha * arr2[k] + (1 - alpha) * arr1[k]
    return res

@jit(float64[:](float64[:],float64),nopython=True)
def mutate(arr,sigma):
    for i in range(arr.shape[0]):
        mutation = np.random.normal()*sigma
        if mutation > sigma*1.5:
            mutation = sigma
        if mutation < -sigma*1.5:
            mutation = -sigma
        arr[i] = arr[i] + mutation
    return arr

@jit(float64(float64[:],float64[:]),nopython=True)
def calc_err(exposure,target):
    err = 0.0
    for i in range(len(exposure)):
        if (target[i] >= 300) and (exposure[i] < 300):
        #if (target[i] >=1) and (exposure[i] < 0.5):
            err += abs(target[i] - exposure[i])
        if (target[i] < 300) and (exposure[i] >= 295):
        #if (target[i] < 1) and (exposure[i] > 0.3):
            err += abs(target[i] - exposure[i])
    return err


@jit(float64[:](float64[:,:],float64[:,:,:]),nopython=True)
def calc_fitness(population,proximity):
    fitness = np.zeros(population.shape[1],dtype=np.float64)
    exposure = np.zeros(population.shape[1],dtype=np.float64)
    pixel_area =  1 #nm^2 #pixel_area * 1e-14  # cm^2

    for p in range(population.shape[1]):
        for j in range(proximity.shape[1]):
            exposure = calc_map(proximity[:,j,:],population[:, p] * current * dwell_time)
            exposure = (exposure* 1e6)/(pixel_area*1e-14 ) # uC/cm^2
            fitness[p] += np.abs(300-np.mean(exposure))
            #fitness[j] = calc_err(exposure, target[])
        fitness[p] = fitness[p]/proximity.shape[1]
    return fitness

@jit(float64[:,:](float64[:,:]),nopython=True)
def recombine_population(population):
    #n_recombination = 6
    #n_recombination = int(population.shape[1]/3)
    n_recombination = int(population.shape[1]/2)

    for i in range(int(n_recombination/2)):
        k = 2*i
        l = 2*i+1
        r_rec = recombine_arrays(population[:, k],population[:, l])
        population[:, -k] = r_rec[:, 0]
        population[:, -l] = r_rec[:, 1]

    return population

@jit(float64[:,:](float64[:,:],float64),nopython=True)
def mutate_population(population,sigma):

    for i in range(population.shape[1]):
        #if i < int(population.shape[1]/3):
        if i < 2:
            population[:, i] = mutate(population[:, i], sigma/20)#
        elif i < 6:
            population[:, i] = mutate(population[:, i], sigma/5)#
        else:
            population[:, i] = mutate(population[:, i], sigma)  #
    return population

@jit(float64[:,:](float64[:,:]),nopython=True)
def check_limits(population):
    for i in range(population.shape[1]):
        for j in range(population.shape[0]):
            if population[j, i] < 1:
                population[j, i] = 0
    return population


population_size = 100 #60
max_iter = 10000

#@jit#(float64(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]))
def iterate(x0,y0,repetitions,target):
    logpoints = np.arange(100,max_iter,100)
    population = np.zeros((len(x0),population_size),dtype=np.float64)
    fitness = np.zeros(population_size,dtype=np.float64)


    proximity = np.zeros((target.shape[0],target.shape[0],target.shape[1]),dtype=np.float64)
    convergence = np.zeros(max_iter)
    t = np.zeros(max_iter)

    for i in range(target.shape[0]):
        for j in range(target.shape[0]):
            for k in range(target.shape[1]):
                proximity[i,j,k] = calc_prox(dist(x0[i],y0[i],target[j,k,0],target[j,k,1]))

    for i in range(population_size):
        population[:, i] = repetitions#+np.random.randint(-50,50)

    print("population initialized")

    starttime = time.time()
    for i in range(max_iter):
        if i < (1/4*max_iter):
            sigma = 2
        else:
            sigma = 0.01 + (1-(i-1/4*max_iter)/(3/4*max_iter))*1


        #sigma = 5 + (1-i/max_iter))*45

        #population = mutate_population(population,sigma)
        #population = check_limits(population)
        fitness = calc_fitness(population,proximity)
        sorted_ind = np.argsort(fitness)
        population = population[:,sorted_ind]
        population = recombine_population(population)
        population = mutate_population(population,sigma)
        population = check_limits(population)
        if i in logpoints:
            print(str(i)+ ": " + str(fitness[sorted_ind][0]))
            #print(sigma)

        convergence[i] = fitness[sorted_ind][0]
        t[i] = time.time() - starttime
        #print(str(i)+ ": " + str(fitness[sorted][0]))

    return population[:,sorted[0]], t, convergence





Outputfile = open(outfilename,'w')

for l in range(len(structures)):
    for k in range(len(dists)):
        print((l,k))

        Outputfile.write('D ' + prefixes[l] +'-'+str(dists[k])+", 11500, 11500, 5, 5" + '\n')
        Outputfile.write('I 1' + '\n')
        Outputfile.write('C '+str(int(dwell_time*1e9)) + '\n')
        Outputfile.write("FSIZE 20 micrometer" + '\n')
        Outputfile.write("UNIT 1 micrometer" + '\n')

        x0,y0 = structures[l](dists[k],radius)
        #randind = np.random.permutation(len(x0))
        #x0 = x0[randind]
        #y0 = y0[randind]

        repetitions = np.ones(len(x0),dtype=np.float64)*100


        x_c,y_c = get_circle(dose_check_radius,10,True)
        #x_t = np.zeros(0,dtype=np.float64)
        #y_t = np.zeros(0, dtype=np.float64)
        target = np.zeros((len(x0),len(x_c),2),dtype=np.float64)
        for i in range(len(x0)):
            target[i,:,0] = x_c+x0[i]
            target[i, :, 1] = y_c + y0[i]

        # fig = plt.figure()
        # plt.imshow(target.reshape(target_shape),extent=[np.min(x_t),np.max(x_t),np.min(y_t),np.max(y_t)])
        # plt.scatter(x0, y0, c="blue")
        # plt.show()
        # plt.close()


        repetitions, t, convergence = iterate(x0, y0, repetitions,target)

        x = np.linspace(np.min(x0)-50,np.max(x0)+50,500)
        y = x
        x, y = np.meshgrid(x, y)
        orig_shape = x.shape
        x = x.ravel()
        y = y.ravel()
        pixel_area = np.abs(x[0] - x[1]) * np.abs(x[0] - x[1])  # nm^2
        pixel_area = pixel_area * 1e-14  # cm^2
        exposure = calc_map_2(x0, y0, repetitions * dwell_time * current, x, y) # C
        exposure = exposure.reshape(orig_shape)
        exposure = exposure * 1e6 # uC

        exposure = exposure/pixel_area # uC/cm^2
        print(np.max(exposure))

        name = "pics/"+prefixes[l]+"_"+str(dists[k])+".png"
        fig = plt.figure()
        cmap = sns.cubehelix_palette(light=1, as_cmap=True,reverse=False)
        plot = plt.imshow(exposure,cmap=cmap,extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
        plt.colorbar()
        plt.contour(x.reshape(orig_shape), y.reshape(orig_shape), exposure.transpose(), [300])#[290,300, 310])
        #plt.scatter(x_t,y_t,c="red")
        #plt.show()
        plt.xlabel('x/nm')
        plt.ylabel('y/nm')
        plt.savefig(name)
        plt.close()

        name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_expected.png"
        fig = plt.figure()
        plot = plt.imshow((exposure >= 300).transpose(),extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
        #plt.contour(x_t.reshape(target_shape), y_t.reshape(target_shape), target.reshape(target_shape), [299],color="black")
        #plt.scatter(x_t,y_t,c="red")
        plt.scatter(x0,y0,c="blue")
        plt.xlabel('x/nm')
        plt.ylabel('y/nm')
        plt.savefig(name)
        plt.close()

        name = "pics/"+prefixes[l]+"_"+str(dists[k])+"_convergence.png"
        fig = plt.figure()
        print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
        plt.semilogy(t,convergence)
        plt.xlabel('time/s')
        plt.ylabel('error')
        plt.savefig(name)
        plt.close()

        #plt.scatter(x_t,y_t,c="red")
        #plt.scatter(x0,y0,c="blue")
        #plt.show()

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














































#
#
# x0 = np.linspace(10,30,num=7)
# y0 = x0
# x0,y0 = np.meshgrid(x0,y0)
# x0 = x0.ravel()
# y0 = y0.ravel()
#
# #x0 = np.linspace(10,30,num=7)
# #y0 = x0*0+20
#
# repetitions = np.ones(len(x0),dtype=np.float64)*200
#
# xt = np.linspace(5,35,50,dtype=np.float64)
# yt = xt
# xt,yt = np.meshgrid(xt,yt)
# shape_t = xt.shape
# xt = xt.ravel()
# yt = yt.ravel()
#
# #target = np.ones(len(xt),dtype=np.float64)
# target = np.zeros(len(xt),dtype=np.float64)
# mask = (xt > 8) & (xt < 32) & (yt > 8) & (yt < 32)
# #mask = (xt > 7) & (xt < 33) & (yt > 17) & (yt < 23)
# target[mask] = 1
#
# # xt = np.linspace(8,32,20,dtype=np.float64)
# # yt = np.ones(20,dtype=np.float64)*8
# # xt = np.append(xt,np.ones(20,dtype=np.float64)*32)
# # yt = np.append(yt,np.linspace(8,32,20,dtype=np.float64))
# # xt = np.append(xt,np.linspace(32,8,20,dtype=np.float64))
# # yt = np.append(yt,np.ones(20,dtype=np.float64)*32)
# # xt = np.append(xt,np.ones(20,dtype=np.float64)*8)
# # yt = np.append(yt,np.linspace(32,8,20,dtype=np.float64))
# #
# # target = np.ones(len(xt),dtype=np.float64)
# #
# # xt = np.append(xt,np.linspace(1,39,20,dtype=np.float64))
# # yt = np.append(yt,np.ones(20,dtype=np.float64)*1)
# # xt = np.append(xt,np.ones(20,dtype=np.float64)*39)
# # yt = np.append(yt,np.linspace(1,39,20,dtype=np.float64))
# # xt = np.append(xt,np.linspace(39,1,20,dtype=np.float64))
# # yt = np.append(yt,np.ones(20,dtype=np.float64)*39)
# # xt = np.append(xt,np.ones(20,dtype=np.float64)*1)
# # yt = np.append(yt,np.linspace(39,1,20,dtype=np.float64))
# #
# # target = np.append(target,np.zeros(len(xt/2),dtype=np.float64))
#
#
# exposure = calc_map_2(x0,y0,repetitions,xt,yt)
#
# res, t, convergence = iterate(x0,y0,repetitions,xt,yt,target)
#
# print(np.array(np.round(res),dtype=np.int))
#
#
# x = np.linspace(0,40,200,dtype=np.float64)
# y = x
# x,y = np.meshgrid(x,y)
# shape1 = x.shape
# x = x.ravel()
# y = y.ravel()
# exposure1 = calc_map_2(x0,y0,res,x,y)
#
# colormap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)
#
# print("time for iteration: "+ str(np.round(np.max(t),2))+" seconds")
# #plt.plot(t[10:],convergence[10:])
# plt.semilogy(t,convergence)
# plt.show()
#
#
# plt.imshow(target.reshape(shape_t),extent=(np.min(xt),np.max(xt),np.min(yt),np.max(yt)))
# plt.scatter(xt,yt,c="red")
# plt.scatter(x0,y0,c="blue")
# plt.show()
#
# plt.imshow(exposure.reshape(shape_t),cmap=colormap,extent=(np.min(xt),np.max(xt),np.min(yt),np.max(yt)))
# plt.colorbar()
# plt.contour(xt.reshape(shape_t), yt.reshape(shape_t), exposure.reshape(shape_t),[0.8,0.9,1])
# plt.show()
#
# plt.imshow(exposure1.reshape(shape1),cmap=colormap,extent=(np.min(x),np.max(x),np.min(y),np.max(y)))
# plt.colorbar()
# plt.scatter(x0,y0)
# plt.contour(xt.reshape(shape_t), yt.reshape(shape_t), target.reshape(shape_t),[0.99],colors=["black"])
# plt.contour(x.reshape(shape1), y.reshape(shape1), exposure1.reshape(shape1),[0.8,0.9,1])
# plt.show()
#
# #plt.imshow(exposure1.reshape(shape1) > 0.9,extent=(np.min(x),np.max(x),np.min(y),np.max(y)))
# #plt.scatter(x0,y0)
# #plt.show()
#
#


