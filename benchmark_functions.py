# benchmark functions
import math
import numpy as np
import os


# global min at 0. 
def spherical(x):

    # xj ∈ [−5.12, 5.12]

    f = np.dot(x, x) 

    return f



# global min found at 0.
def ackley(x):

    # xj ∈ [−32.768, 32.768]

    n = len(x)
    f = -20*np.exp(-0.2*np.sqrt((1/n)*np.dot(x, x))) - np.exp((1/n)*np.sum(np.cos(2*np.pi*x))) + 20 + math.e 

    return f


# global min at -19.6370136
def michalewicz(x):

    # xj ∈ [0, π]

    m = 10
    terms = [np.sin(x[j])*(np.sin((j*(x[j]**2))/np.pi))**(2*m) for j in range(len(x))]
    f = - np.sum(terms)

    return f



# global min at 0
def katsuura(x):


    # xj ∈ [0, 100]


    """n = len(x)
    product = 1
    for i in range(n):

        summation = [(np.abs(2**j*x[i]- np.round(2**j*x[i]))/2**j) for j in range(1, 33)] # range from 1 to 32
        term = 1 + i*np.sum(summation)
        term = term**(10/(n**1.2))
        product = product*term

    f = (10/n)*product - (10/(n**2)) """
    x = 0.05 * x 
    nx = len(x)
    pw = 10/(nx**1.2)
    prd = 1.0
    tj = 2**np.arange(start=1, stop=33, step=1)
    for i in range(0, nx):
        tjx = tj*x[i]
        t = np.abs(tjx - np.round(tjx)) / tj
        tsm = 0.0
        for j in range(0, 32):
            tsm += t[j]
        prd *= (1+ (i+1)*tsm)**pw
    df = 10/(nx*nx)
    f = df*prd - df

    return f



def shubert(x):

    # xj ∈ [−10, 10]

    #product = 1

    sums = []
    for j in range(len(x)):
    
        summation = np.sum([i*np.cos((i+1)*x[j]+i) for i in range(1, 6)]) # must go from 1 to 5
        #print(summation)
        sums.append(summation)
        #product = product*summation

    product = np.prod(sums)
    f = product

    return f





# global min found at 500
# rotated and shifted ackley
def r_s_ackley(x, M, o):

    # xj ∈ [−32.768, 32.768]

    z = np.matmul(M, x-o)
    n = len(z) 
    f = -20*np.exp(-0.2*np.sqrt((1/n)*np.dot(z, z))) - np.exp((1/n)*np.sum(np.cos(2*np.pi*z))) + 20 + math.e # sum of squares is the dot product??
    final_f = f + 500

    return final_f



# global min at -19.6370136 + 500
# rotated and shifted michalewicz
def r_s_michalewicz(x, M, o):

    # xj ∈ [−32.768, 32.768]

    z = np.matmul(M, x-o)
    n = len(z) 
    m = 10
    terms = [np.sin(z[j])*(np.sin((j*(z[j]**2))/np.pi))**(2*m) for j in range(len(z))]
    f = - np.sum(terms)
    final_f = f + 500 

    return final_f



# global min at 500
def r_s_katsuura(x, M , o):


    # xj ∈ [0, 100]

    z = np.matmul(M, x-o)
    n = len(z) # check if correct

    z = 0.05 * z # where is this step in the equation??
    nz = len(z)
    pw = 10/(nz**1.2)
    prd = 1.0
    tj = 2**np.arange(start=1, stop=33, step=1)
    for i in range(0, nz):
        tjz = tj*z[i]
        t = np.abs(tjz - np.round(tjz)) / tj
        tsm = 0.0
        for j in range(0, 32):
            tsm += t[j]
        prd *= (1+ (i+1)*tsm)**pw
    df = 10/(nz*nz)
    f = df*prd - df

    final_f = f + 500 

    return final_f



def r_s_shubert(x, M, o):

    # xj ∈ [−10, 10]

    z = np.matmul(M, x-o)
    n = len(z) 

    product = 1

    for j in range(len(z)):
    
        summation = np.sum([i*np.cos((i+1)*z[j]+i) for i in range(1, 6)]) # must go from 1 to 5
        product = product*summation

    f = product
    final_f = f + 500 

    return final_f    