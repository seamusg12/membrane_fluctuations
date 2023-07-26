##############################
# fourier.py                 #
# Seamus Gallagher           #
# Carnegie Mellon University #
##############################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import sys

g = int(sys.argv[1])
path = str(g)+"/"

def average_empties(arr,ctr_arr): # empty (real space) grid cell ---> average of 4 nearest neighbors
    ind_list = np.argwhere(ctr_arr==0)
    if (len(ind_list) == 0): return
    for ind in ind_list:
        n1 = tuple((ind + np.array([0,1,0]))%np.array([np.inf,g,g]))
        n2 = tuple((ind + np.array([0,-1,0]))%np.array([np.inf,g,g]))
        n3 = tuple((ind + np.array([0,0,1]))%np.array([np.inf,g,g]))
        n4 = tuple((ind + np.array([0,0,-1]))%np.array([np.inf,g,g]))
    # Convert floats to ints (made float from using np.inf)
        n1 = tuple(int(x) for x in n1)
        n2 = tuple(int(x) for x in n2)
        n3 = tuple(int(x) for x in n3)
        n4 = tuple(int(x) for x in n4)
        
        neighbors = np.array([arr[n1],arr[n2],arr[n3],arr[n4]])
        neighbors_count = np.array([ctr_arr[n1],ctr_arr[n2],ctr_arr[n3],ctr_arr[n4]])

        arr[tuple(ind)] = np.sum(neighbors)/len(neighbors[neighbors_count!=0])
        ctr_arr[tuple(ind)] = np.sum(neighbors_count)/len(neighbors_count[neighbors_count!=0])

    return

def wave_numbers(Lavg,g): # labels q vectors according to numpy fft logic (i.e. in 1D with 6 modes order goes 0 1 2 3 -2 -1)
    q2_arr = np.zeros((g,g)) # also respects that input is real so ~ half the output array is extraneous
    qx_arr = np.zeros((g,g))
    qy_arr = np.zeros((g,g))
    for i in range(g):
        for j in range(g):
            if i+j==0: continue
            if i == 0:
                if j > g/2:
                    qx_arr[i,j] = np.nan
                    qy_arr[i,j] = np.nan
                    q2_arr[i,j] = np.nan
            if i > g/2:
                qx_arr[i,j] = np.nan
                qy_arr[i,j] = np.nan
                q2_arr[i,j] = np.nan
            qx=(i-g*(i>g/2))*2*np.pi/Lavg
            qy=(j-g*(j>g/2))*2*np.pi/Lavg
            qx_arr[i,j] = qx
            qy_arr[i,j] = qy
            q2_arr[i,j]= qx**2+qy**2

    return q2_arr,qx_arr,qy_arr

def block(arr): 
    sig_list = [np.nanstd(arr,axis=0)/np.sqrt(len(arr))]
    dsig_list =[sig_list[-1]/np.sqrt(2*(len(arr)-1))]
    c = np.copy(arr)
    while(len(c) > 2):
        temp = []
        for i in range(0,len(c)-1,2):
            temp.append( (c[i]+c[i+1])/2 )
        c = np.array(temp)
        sig_list.append(np.nanstd(c,axis=0)/np.sqrt(len(c)))
        dsig_list.append(sig_list[-1]/np.sqrt(2*(len(c)-1)))
    return np.array(sig_list), np.array(dsig_list)

def FT(head,tail,surf,Lt,g):
    Lavg = np.mean(Lt) # get wave numbers first
    q2_arr,qx_arr,qy_arr = wave_numbers(Lavg,g)


    N = len(head[0]) # number of lipids
    n = head - tail
    norms = np.abs(n[:,:,2]) # z-normalization
    #norms =  np.sum((tail-head)**2,axis=2)**.5 # L normalization

    n /= norms[:,:,None]

    h_agg = np.zeros((T,g,g)) # heights grid
    h_agg1 = np.zeros((T,g,g)) # leaflet 1
    h_agg2 = np.zeros((T,g,g)) # leaflet 2

    n_agg = np.zeros((T,g,g,2)) # directors grid
    n_agg1 = np.zeros((T,g,g,2))
    n_agg2 = np.zeros((T,g,g,2))

    counts_h1 = np.zeros((T,g,g)) # raw counts for grid spots
    counts_h2 = np.zeros((T,g,g))
    counts_n1 = np.zeros((T,g,g))
    counts_n2 = np.zeros((T,g,g))


    print("Mapping surface and director fields")
    start = time.time()
    for t in range(T):
        if t % 100 == 0:
            print(str(t) + " / " + str(T))
        L = Lt[t]
        z_avg = np.mean(surf[t,:,2]%40) #  THIS NEEDS TO BE CHANGED FOR ASYMMETRIC MEMBRANES
        for l in range(N):
            i = int((surf[t,l,0]%L) / (L/g)) # grid indices for given lipid
            j = int((surf[t,l,1]%L) / (L/g))

            if head[t,l,2] > z_avg:
                h_agg1[t,i,j] += surf[t,l,2]
                counts_h1[t,i,j] += 1
                if np.linalg.norm(n[t,l]) < 10:
                    n_agg1[t,i,j] += n[t,l,:2]
                    counts_n1[t,i,j] += 1
            else:
                h_agg2[t,i,j] += surf[t,l,2]
                counts_h2[t,i,j] += 1
                if np.linalg.norm(n[t,l]) < 10:
                    n_agg2[t,i,j] += n[t,l,:2]
                    counts_n2[t,i,j] += 1

    global su
    del su,surf
    end = time.time()
    #print("Time elapsed mapping fields: " + str(end-start) + " s")

    print("Averaging empty grid sites")
    start = time.time()
    for arrs in [[h_agg1,counts_h1],[h_agg2,counts_h2],[n_agg1,counts_n1],[n_agg2,counts_n2]]:
        average_empties(arrs[0],arrs[1])
    end = time.time()
    #print("Time elapsed averaging empty grid sites: " + str(end-start) + " s")

    # Average leaflets
    h_agg = (h_agg1/counts_h1 + h_agg2/counts_h2)/2
    n_agg = (n_agg1/counts_n1[:,:,:,None] - n_agg2/counts_n2[:,:,:,None])/2
    n_agg_bar = (n_agg1/counts_n1[:,:,:,None] + n_agg2/counts_n2[:,:,:,None])/2
    print("FFT")
    # FFT 
    hq_agg = np.fft.fft2(h_agg,axes=(1,2))*Lt[:,None,None]/g/g 
    nq_agg = np.fft.fft2(n_agg,axes=(1,2))*Lt[:,None,None,None]/g/g
    nq_agg_bar = np.fft.fft2(n_agg_bar,axes=(1,2))*Lt[:,None,None,None]/g/g
    # Normalize
    print("Decomposing directors")
    # Decompose directors into parallel and transverse components
    nqll_agg = np.zeros((T,g,g),dtype=complex)
    nqL_agg = np.zeros((T,g,g),dtype=complex)

    nqll_agg_bar = np.zeros((T,g,g),dtype=complex)
    nqL_agg_bar = np.zeros((T,g,g),dtype=complex)

    for i in range(g):
        for j in range(g):
            if i + j == 0:
                nqll_agg[:,i,j] = np.nan
                nqL_agg[:,i,j] = np.nan
                hq_agg[:,i,j] = np.nan
                nqll_agg_bar[:,i,j] = np.nan
                nqL_agg_bar[:,i,j] = np.nan
                continue
            q = np.array([qx_arr[i,j],qy_arr[i,j]])
            qhat = q/np.linalg.norm(q)
            qhatL = np.array([qhat[1],-qhat[0]])
            nqll_agg[:,i,j] = np.dot(nq_agg[:,i,j],qhat)
            nqL_agg[:,i,j] = np.dot(nq_agg[:,i,j],qhatL)
            nqll_agg_bar[:,i,j] = np.dot(nq_agg_bar[:,i,j],qhat)
            nqL_agg_bar[:,i,j] = np.dot(nq_agg_bar[:,i,j],qhatL)

    # return coefficients squared
    hq2 = np.abs(hq_agg)**2
    nqll2 = np.abs(nqll_agg)**2
    nqL2 = np.abs(nqL_agg)**2
    nqll2_bar = np.abs(nqll_agg_bar)**2
    nqL2_bar = np.abs(nqL_agg_bar)**2
    return hq2,nqll2,nqL2,nqll2_bar,nqL2_bar,q2_arr

h = np.load('head.npy',allow_pickle=True)
t = np.load('tail.npy',allow_pickle=True)
su = np.load('surf.npy',allow_pickle=True)
si = np.load('size.npy',allow_pickle=True)

T = len(h)
hq2,nqll2,nqL2,nqll2_bar,nqL2_bar,q2 = FT(h,t,su,si,g)
q = np.sqrt(q2)

y1 = np.ravel(np.nanmean(hq2,axis=0)*np.power(q,4))
y2 = np.ravel(np.nanmean(nqll2,axis=0)*np.power(q,2))
y3 = np.ravel(np.nanmean(nqL2,axis=0))
y4 = np.ravel(np.nanmean(nqll2_bar,axis=0))
y5 = np.ravel(np.nanmean(nqL2_bar,axis=0))

b1,db1 = block(hq2)
b2,db2 = block(nqll2)
b3,db3 = block(nqL2)
b4,db4 = block(nqll2_bar)
b5,db5 = block(nqL2_bar)

plt.clf()
plt.errorbar(range(len(b1)), b1[:,1,0], yerr=db1[:,1,0])
plt.savefig(path+"h10_blocking.pdf")
plt.clf()
plt.errorbar(range(len(b1)), b1[:,2,1], yerr=db1[:,2,1])
plt.savefig(path+"h21_blocking.pdf")
plt.clf()


with open(path+"blocking10.txt","w") as f: # Save some blocking curves for different modes
    for i in range(len(b1)):
        f.write( str(b1[i,1,0]) + " " + str(b2[i,1,0]) + " " + str(b3[i,1,0]) + " " + str(b4[i,1,0]) + " " + str(b5[i,1,0]) + " " + str(db1[i,1,0]) + " " + str(db2[i,1,0]) + " " + str(db3[i,1,0]) + " " + str(db4[i,1,0]) + " " + str(db5[i,1,0]) + "\n" )
with open(path+"blocking21.txt","w") as f:
    for i in range(len(b1)):
        f.write( str(b1[i,2,1]) + " " + str(b2[i,2,1]) + " " + str(b3[i,2,1]) + " " + str(b4[i,2,1]) + " " + str(b5[i,2,1]) + " " + str(db1[i,2,1]) + " " + str(db2[i,2,1]) + " " + str(db3[i,2,1]) + " " + str(db4[i,2,1]) + " " + str(db5[i,2,1]) + "\n" )

dy1 = np.zeros((g,g))
dy2 = np.zeros((g,g))
dy3 = np.zeros((g,g))
dy4 = np.zeros((g,g))
dy5 = np.zeros((g,g))

for i in range(g):
    for j in range(g):
        dy1[i,j] = b1[np.argmax(b1[:,i,j]),i,j]
        dy2[i,j] = b2[np.argmax(b2[:,i,j]),i,j]
        dy3[i,j] = b3[np.argmax(b3[:,i,j]),i,j]
        dy4[i,j] = b4[np.argmax(b4[:,i,j]),i,j]
        dy5[i,j] = b5[np.argmax(b5[:,i,j]),i,j]
dy1 = np.ravel(dy1*np.power(q,4))
dy2 = np.ravel(dy2*np.power(q,2))
dy3 = np.ravel(dy3)
dy4 = np.ravel(dy4)
dy5 = np.ravel(dy5)

q = np.ravel(q)
sort_ind = np.argsort(q)

q = q[sort_ind]

y1 = y1[sort_ind]
y2 = y2[sort_ind]
y3 = y3[sort_ind]
y4 = y4[sort_ind]
y5 = y5[sort_ind]

dy1 = dy1[sort_ind]
dy2 = dy2[sort_ind]
dy3 = dy3[sort_ind]
dy4 = dy4[sort_ind]
dy5 = dy5[sort_ind]

np.save(path+'q.npy',q)

np.save(path+'y1.npy',y1)
np.save(path+'y2.npy',y2)
np.save(path+'y3.npy',y3)
np.save(path+'y4.npy',y4)
np.save(path+'y5.npy',y5)

np.save(path+'dy1.npy',dy1)
np.save(path+'dy2.npy',dy2)
np.save(path+'dy3.npy',dy3)
np.save(path+'dy4.npy',dy4)
np.save(path+'dy5.npy',dy5)
