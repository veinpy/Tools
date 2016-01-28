import numpy as np
import random
from numpy import zeros
from sklearn.cluster import KMeans
import copy

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X,ks=range(1,10)):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    Wks = zeros(len(ks))
    Wkbs = zeros(len(ks))
    sk = zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sdk = copy.deepcopy(sk)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk, sdk)

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))


def find_centers(X, K):
    # Initialize to K random centers
    model = KMeans(n_clusters=K,max_iter=500,n_jobs=6)
    model.fit(X)
    #print 'fit K=%s finished' %K
    mu = model.cluster_centers_
    new_mu = []
    for i in range(len(mu)):
        new_mu.append(mu[i,:])
    mu=new_mu
    clusters_label = model.labels_
    label_dict={}
    for i,j in enumerate(clusters_label):
        if j not in label_dict:
            label_dict[j] = [X[i,:]]
        else:
            label_dict[j].append(X[i,:])
    clusters= []
    for i in sorted(label_dict.keys()):
        clusters.append(label_dict[i])
    return mu,clusters

def determine_k(*args):
    assert len(args)==4
    ks = args[0];logWks = args[1];logWkbs=args[2];sk=args[3]
    K= []
    for i in range(len(ks)):
        if logWkbs[i]-logWks[i]>=(logWkbs[i+1]-logWks[i+1])-sk[i+1]:
            K.append(ks[i])   

def K_determin(X,plot=False):
    ks, logWks, logWkbs, sk,sdk  = gap_statistic(X)
    gaps = []
    for i in range(len(ks)):
        gaps.append(logWkbs[i]-logWks[i])
        if logWkbs[i]-logWks[i]>=(logWkbs[i+1]-logWks[i+1])-sk[i+1]:
            print ks[i]
    if plot:
        from matplotlib import pyplot as plt
        plt.plot(gaps)
        plt.title('gaps')
        plt.show()
    return ks

if __name__ =='__main__':
    X = init_board_gauss(200,3)
    ks, logWks, logWkbs, sk,sdk = gap_statistic(X)
    for i in range(len(ks)):
        if logWkbs[i]-logWks[i]>=(logWkbs[i+1]-logWks[i+1])-sk[i+1]:
            print ks[i]
