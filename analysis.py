##################################################################################################
# analysis.py
# Seamus Gallagher
# Carnegie Mellon University
##################################################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import sys

##################################################################################################
# Analysis functions
##################################################################################################
def red_chi_sq(y_o,y_c,sigma,dof): 
    X2 = 0
    for i,y in enumerate(y_o):
        X2 += (y - y_c[i])**2/(sigma[i]**2)
    return X2/dof

def compute_qmax(f,q,y,dy,dof,p0=None,bounds=[-np.inf,np.inf],absolute_sigma=True): # Calculate chisquared as a function of maximum q (theory must break down at high q)
    X2_list = []
    min_ind = 5 # where to start chi square fitting so youre not just fitting 1 or two pts
    
    for max_ind in range(min_ind,len(q)):
        try:
            popt,pcov = curve_fit(f,q[1:max_ind],y[1:max_ind],sigma=dy[1:max_ind],p0=p0,bounds=bounds,absolute_sigma=absolute_sigma)
            X2 = red_chi_sq(y[1:max_ind],[f(*(x,*popt)) for x in q[1:max_ind]],dy[1:max_ind],max_ind-dof-1) # this might not work with the unpacking operator
        except RuntimeError: 
            print("Error - curve_fit failed")
            X2_list.append(np.inf)

    qmax = np.argmin(np.abs(np.array(X2_list)-1)) + min_ind # should change the maximum to something more prosaic

    return qmax

def bootstrap(f,q,y,dy,p0=None,bounds=[-np.inf,np.inf],absolute_sigma=True,NBoot=1000):
    arr = []

    for i in range(Nboot):
        ytest = np.random.normal(y,dy,len(y))
        try: 
            popt,pcov = curve_fit(f_h,q,ytest,sigma=dy,p0=p0,bounds=bounds,absolute_sigma=absolute_sigma)
            arr.append(np.array(popt))
        except RuntimeError: print("Error - curve_fit failed")

    return np.mean(np.array(arr),axis=0),np.std(np.array(arr),axis=0)

##################################################################################################
# Fitting functions
##################################################################################################

def f_h(q,qc,k,kteff):
    return (1/k + (q**2)/kteff)/(1-(q/qc)**2)

def f_nL(q,ktw,kt): # also the function for the symmetrized directors
    return 1/(kt+ktw*(q**2))

def f_nll(q,qc,k):
    return (1/k)/(1-(q/qc)**2 )

def f_nll_bar(q,kbd,kt,kq):
    return 1/(kt+kbd*pow(q,2)+kq*pow(q,4))

##################################################################################################
# Main analysis script
##################################################################################################

q = np.load('q.npy')

y1 = np.load('y1.npy')
y2 = np.load('y2.npy')
y3 = np.load('y3.npy')
y4 = np.load('y4.npy')
y5 = np.load('y5.npy')

dy1 = np.load('dy1.npy')
dy2 = np.load('dy2.npy')
dy3 = np.load('dy3.npy')
dy4 = np.load('dy4.npy')
dy5 = np.load('dy5.npy')


qmaxh = compute_qmax(f_h,q,y1,dy1,3,p0=[3,20,5])
qmaxll = compute_qmax(f_nll,q,y2,dy2,2,p0=[3,20])
qmaxL = compute_qmax(f_nL,q,y3,dy3,2,p0=[5,5])
qmaxll_bar = compute_qmax(f_nll_bar,q,y4,dy4,3)
qmaxL_bar = compute_qmax(f_nL,q,y5,dy5,2,p0=[5,5])

popt_h,dpopt_h = bootstrap(f_h,q,y1,dy1)
popt_nll,dpopt_nll = bootstrap(f_nll,q,y2,dy2)
popt_nL,dpopt_nL = bootstrap(f_nL,q,y3,dy3)
popt_nll_bar,dpopt_nll_bar = bootstrap(f_nll_bar,q,y4,dy4)
popt_nL_bar,dpopt_nL_bar = bootstrap(f_nL,q,y5,dy5)

with open("parameters.txt","w") as f:
    f.write(" ".join([str(x) for x in popt_h])+"\n")
    f.write(" ".join([str(x) for x in dpopt_h])+"\n")
    f.write(" ".join([str(x) for x in popt_nll])+"\n")
    f.write(" ".join([str(x) for x in dpopt_nll])+"\n")
    f.write(" ".join([str(x) for x in popt_nL])+"\n")
    f.write(" ".join([str(x) for x in dpopt_nL])+"\n")
    f.write(" ".join([str(x) for x in popt_nll_bar])+"\n")
    f.write(" ".join([str(x) for x in dpopt_nll_bar])+"\n")
    f.write(" ".join([str(x) for x in popt_nL_bar])+"\n")
    f.write(" ".join([str(x) for x in dpopt_nL_bar])+"\n")
    

##################################################################################################
# Plotting
##################################################################################################

xplot = [0]+list(q)
fig = plt.figure(figsize=(20,5))

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

ax1.plot(xplot,[f_h(x,popt_h[0],popt_h[1],popt_h[2]) for x in xplot])
ax2.plot(xplot,[f_nll(x,popt_nll[0],popt_nll[1]) for x in xplot])
ax3.plot(xplot,[f_nL(x,popt_nL[0],popt_nL[1]) for x in xplot])

ax1.errorbar(q,y1,yerr=dy1,fmt='o')
ax1.set_ylabel("$q^4|h_{q}|^2$ ($\epsilon$)")
ax1.set_xlabel("$q$($\sigma^{-1}$)")
ax1.set_xlim(left=0)
ax1.set_ylim((0.01,3))

ax2.errorbar(q,y2,yerr=dy2,fmt='o')
ax2.set_ylabel("$q^2|n_{q\parallel}|^2$ ($\epsilon/\sigma^{2}$)")
ax2.set_xlabel("$q$($\sigma^{-1}$)")
ax2.set_xlim(left=0)
ax2.set_ylim((0.01,0.5))

ax3.errorbar(q,y3,yerr=dy3,fmt='o')
ax3.set_ylabel("$|n_{q\perp}|^2$ ($\epsilon$)")
ax3.set_xlabel("$q$($\sigma^{-1}$)")
ax3.set_xlim(left=0)


plt.savefig('antisymmetrized.pdf')

fig = plt.figure(figsize=(20,5))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.plot(xplot,[f_nll_bar(x,popt_nll_bar[0],popt_nll_bar[1],popt_nll_bar[2]) for x in xplot])
ax2.plot(xplot,[f_nL(x,popt_nL_bar[0],popt_nL_bar[1]) for x in xplot])

ax1.errorbar(q,y4,yerr=dy4,fmt='o')
ax1.set_ylabel("$|n_{+q\parallel}|^2$ ($\epsilon$)")
ax1.set_xlabel("$q$($\sigma^{-1}$)")
ax1.set_xlim(left=0)

ax2.errorbar(q,y5,yerr=dy5,fmt='o')
ax2.set_ylabel("$|n_{+q\perp}|^2$ ($\epsilon$)")
ax2.set_xlabel("$q$($\sigma^{-1}$)")
ax2.set_xlim(left=0)

plt.savefig('symmetrized.pdf')
