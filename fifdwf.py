import sys
import cmath
import scipy
import quadpy
import numpy as np
import matplotlib.pyplot as plt


a = .5
b = 2
infinity = 5
timeSteps = 1024
period = 1
terms = 50
time = np.linspace(0,period,num=timeSteps)
maxCoefs = terms*infinity*(2**infinity)

values = [0, 0, 0.25, .6, 0.5, .25, 0.75, 1, 1, 0]
timeSteps = 1000
globalIt = 3

d = [-.5, .5, -.5, .5]
a0 = []
e = []
c = []
f = []
nMaps = int((len(values)/2)-1)

def cn(n):
   c = z*np.exp(-1j*2*n*np.pi*time/period)
   return c.sum()/c.size

def f0(x, terms):
   f = np.array([2*cn(i)*np.exp(1j*2*i*np.pi*x/period) for i in range(1,terms)])
   return cn(0)+f.sum()

def ek(k,x):
   if (k == 0):
      return 1
   elif (k % 2 != 0):
      y = np.array([(a**m)*cmath.exp(2*np.pi*1j*k*(2**m)*x) for m in range(infinity)])
      return np.sqrt(1-(a**2)) * y.sum()
   elif (k % 2 == 0): 
      y = np.array([(a**m)*cmath.exp(2*np.pi*1j*k*(2**m)*x) for m in range(infinity)])
      x = np.array([(a**m)*cmath.exp(2*np.pi*1j*int(k/2)*(2**m)*x) for m in range(infinity)])
      return ((np.sqrt(1-(a**2)) * y.sum())-(a*(np.sqrt(1-(a**2)) * x.sum()))) / np.sqrt(1-a**2)

def ak(k):
   if (k == 0):
      return coefs[0]
   elif (k % 2 != 0):
      y = np.array([(a**m)*coefs[k*(2**m)] for m in range(infinity)])
      return np.sqrt(1-(a**2)) * y.sum()
   elif (k % 2 == 0):
      y = np.array([(a**m)*coefs[k*(2**m)] for m in range(infinity)])
      x = np.array([(a**m)*coefs[int(k/2)*(2**m)] for m in range(infinity)])
      return ((np.sqrt(1-(a**2)) * y.sum()) - (a*(np.sqrt(1-(a**2)) * x .sum()))) / np.sqrt(1-a**2)


# FIF
def FIF(x, it):
   if (it == 0):
      p = np.array([(c[i]*((x - e[i]) / a0[i]) + f[i]) * uFunc(x,i)
                     for i in range(nMaps)])
      return p.sum()
   p = np.array([(d[i]*(FIF((x - e[i]) / a0[i], it-1)) +
                  c[i]*((x - e[i]) / a0[i]) + f[i]) * uFunc(x,i)
                  for i in range(nMaps)])
   return p.sum()

# Returns 1 or 0 based on section between interpolation points x is in
def uFunc(x,i):
   if (i+1 > nMaps-1):
      return 1 if x >= e[i] else 0
   u1 = 1 if x - e[i] >= 0 else 0
   u2 = 1 if x - e[i+1] >= 0 else 0
   return u1 - u2

# Used when FIF is expected to be between 0 and 1 with evenly space points
def simpleParameters(values):
   k = 2
   for i in range(nMaps):
      a0.append(1 / nMaps)
      e.append(i / nMaps)
      c.append(values[k+1] - values[k-1])
      f.append(values[k-1])
      k=k+2


def functionGuy(x):
   return FIF(x, globalIt)
	  
def wFourier(x):
   y = np.array([ak(k)*ek(k,x) for k in range(1,terms)])
   return coefs[0]+2*y.sum().real


def main():
    global z
    global coefs
    simpleParameters(values)
    z = np.array([functionGuy(t).real for t in time])
    f = np.array([f0(t, terms).real for t in time])
    plt.plot(time, z)
    plt.plot(time, f)
    coefs = np.array([cn(n) for n in range(maxCoefs)])
    points = np.array([wFourier(t) for t in time])
    plt.plot(time, points)


    print(values)
    print("a:", a)
    print("e:", e)
    print("c:", c)
    print("f:", f)
    print("d:", d)
    plt.show()

def clean():
    print("Exiting")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        clean()
