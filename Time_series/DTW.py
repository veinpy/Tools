#coding:utf-8

"""
Document:
    https://www.evernote.com/shard/s659/nl/125781947/2a5a6a17-35c6-4534-9504-79b593bfa928/?csrfBusterToken=U%3D77f47bb%3AP%3D%2F%3AE%3D152746f86b1%3AS%3Dd5b6eed934dc2892ed872ad07cd12084
"""
from math import sqrt
"""
Speeding Up Dynamic Time Warping

Dynamic time warping has a complexity of O(nm) where n is the length of the first time series and m is the length of the second time series. If you are performing dynamic time warping multiple times on long time series data, this can be prohibitively expensive. However, there are a couple of ways to speed things up. The first is to enforce a locality constraint. This works under the assumption that it is unlikely for qi and cj to be matched if i and j are too far apart. The threshold is determined by a window size w. This way, only mappings within this window are considered which speeds up the inner loop. The following is the modified code which includes the window size w.

"""
def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])

"""
Another way to speed things up is to use the LB Keogh lower bound of dynamic time warping. It is defined as

LBKeogh(Q,C)=∑ni=1(ci−Ui)2I(ci>Ui)+(ci−Li)2I(ci<Li)

where Ui and Li are upper and lower bounds for time series Q which are defined as Ui=max(qi−r:qi+r) and Li=min(qi−r:qi+r) for a reach r and I(⋅) is the indicator function. It can be implemented with the following function.


"""
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return sqrt(LB_sum)

##testing code
import numpy as np

from sklearn.metrics import classification_report

def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
        preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds)

def try_knn():
    train = np.genfromtxt('train.csv', delimiter='\t')
    test = np.genfromtxt('test.csv', delimiter='\t')
    print knn(train,test,4)

# clustering
import random

def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print counter
        assignments={}
        #assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]

        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]

    return centroids

def try_clustering():
    train = np.genfromtxt('train.csv', delimiter='\t')
    test = np.genfromtxt('test.csv', delimiter='\t')
    data=np.vstack((train[:,:-1],test[:,:-1]))

    import matplotlib.pylab as plt

    centroids=k_means_clust(data,4,10,4)
    for i in centroids:

        plt.plot(i)

    plt.show()