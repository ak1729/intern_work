# Importing required libraries
import numpy as np
import math
import time
import pandas as pd
import scipy.linalg
import numpy.linalg
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import umap
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from hyperopt import fmin, tpe, hp


# %matplotlib inline
#from cuml.neighbors import NearestNeighbors
# Axes3D

# defining number of neighbours
nbor = 5  


def localplusglobal(X,Y):
    def delta(A):
        if (A.shape[1]==1):
            B = np.zeros((A.shape[0],A.shape[0]))
            for i in range(A.shape[0]):
                B[i,i] = A[i,0]
        if(A.shape[1]==A.shape[0]):
            B = np.zeros((A.shape[0],1))
            for i in range(A.shape[0]):
                B[i,0]= A[i,i]
        return B
    
#     Gradient of Objective Function

    def gradphi(P_i,M_i):
        P_new = np.dot(P_i.T,M_i)
        P_new = delta(delta(P_new))
        k1 = 2*np.dot(M_i,P_new)
        k2 = np.dot(P_i,P_i.T)
        k3 = np.dot(M_i.T,P_i)
        k4 = np.dot(np.dot(k2,M_i),P_new)
        k5 = np.dot(np.dot(P_i,P_new),k3)
        phi = k1 - k4 - k5
        return phi
    
    #     Adagrad Optimizer Function
    
    def adagrad(Pi, Ai, alpha, eps):
        cnt=0
        lr=alpha
        max_ite=1000
        epsilon=1e-8
        G = np.zeros_like(Pi)

        for it in range(max_ite):
            cnt=cnt+1
            grad_Pi = gradphi(Pi, Ai)
            G += grad_Pi ** 2
            adjusted_lr = lr / (np.sqrt(G) + epsilon)
            Pi += adjusted_lr * grad_Pi
            Q, _ = np.linalg.qr(Pi)
            Pi=Q
            if(np.linalg.norm(grad_Pi)<eps):
                break

        return Pi, cnt
    
    
    #    Adadelta Optimizer Function
    
    def adadelta(Pi, Ai, eps):
        accum_grad_Pi = np.zeros_like(Pi)
        accum_delta_Pi = np.zeros_like(Pi)
        epsilon=1e-8
        rho=0.9
        counter=0
        max_iters=1000

        for iteration in range(max_iters):
            gradient_Pi = gradphi(Pi, Ai)
            accum_grad_Pi = rho * accum_grad_Pi + (1 - rho) * (gradient_Pi ** 2)
            update_Pi = np.sqrt(accum_delta_Pi + epsilon) * gradient_Pi / np.sqrt(accum_grad_Pi + epsilon)
            Pi += update_Pi
            accum_delta_Pi = rho * accum_delta_Pi + (1 - rho) * (update_Pi ** 2)
            counter+=1
            if(np.linalg.norm(gradient_Pi)<eps):
                break

        return Pi, counter
    
    
#     RMSprop Optimizer Function
    
    def rmsprop(Pi, Ai, alpha, eps):
        G = np.zeros_like(Pi)
        epsilon=1e-8
        decay_rate=0.9
        lr=alpha
        counter=0
        max_iters=1000
        for iteration in range(max_iters):
            grad_Pi = gradphi(Pi, Ai)
            G = decay_rate * G + (1 - decay_rate) * np.square(grad_Pi)
            Pi += lr * grad_Pi / (np.sqrt(G) + epsilon)
            Q, _ = np.linalg.qr(Pi)
            Pi=Q
            counter+=1
            if(np.linalg.norm(grad_Pi)<eps):
                break

        return Pi, counter
    
#     MEQA Optimizer
    
    def meqa(Pi, Ai, alpha, eps):
        counter=0
        for it in range(1000):
            grad=gradphi(Pi, Ai)
            Pi += alpha*grad
            Q, _ = np.linalg.qr(Pi)
            Pi=Q
            counter+=1
            if(np.linalg.norm(grad)<eps):
                break
        return Pi, counter

    
    def asim(X,Y,k2):
        ek = np.ones((k2,1))
        ik = np.identity(k2)
        X1 = X
        Y1 = Y
        X = np.dot(X,(ik-(1/k2)*np.dot(ek,ek.T)))
        Y = np.dot(Y,(ik-(1/k2)*np.dot(ek,ek.T)))
        norm_x = (X**2.).sum()
        A = np.dot(X,Y.T)
        U,S,V = np.linalg.svd(A, full_matrices = False)
        P = np.dot(U,V)
        k = delta(delta(np.dot(Y,Y.T)))
        k1 = np.linalg.pinv(scipy.linalg.sqrtm(k))
        m = np.dot(A,k1)
        grad= gradphi(P,m)
        
        
# ****************This section is to select different optimizer******************

#        Max Iteration is set to 1000

#         All the function return optimized P and cnt as number of iteration it take to optimize
#         P, cnt = meqa(P, m, 0.1, 1e-10)   # Calling MEQA
#         P, cnt = adagrad(P, m, 0.1, 1e-10)   # Calling adagrad optimizer function
#         P, cnt=adadelta(P, m, 1e-10)       # Calling adadelta optimizer function
        P, cnt=rmsprop(P, m, 0.1, 1e-10)    # Calling rmsprop optimizer function
        
# *********************************************************************************
    
        
        k3 = np.dot(np.dot(P.T,X),Y.T)
        k4 = delta(delta(k3))
        D = np.dot(np.linalg.pinv(k),k4)
        t = (1/k2)*np.dot((X1-np.dot(np.dot(P,D),Y1)),ek)
        s = 0
        Xtemp = np.zeros((X.shape[0],1))
        Ytemp = np.zeros((Y.shape[0],1))
        for i in range(X.shape[1]):
          Xtemp[:,0] = X1[:,i]
          Ytemp[:,0] = Y1[:,i]
          s = s + (np.linalg.norm((Xtemp-np.dot(np.dot(P,D),Ytemp)-t))**2)/norm_x
        return s,P,D,t, cnt
    
    def nieqa(X,Y):
        X = X.T
        Y = Y.T
        
        knn = NearestNeighbors(n_neighbors=nbor)
        knn.fit(X)
        dist,arr= knn.kneighbors(return_distance = True)
        wtmatrix = (math.inf)*(np.ones((X.shape[0],X.shape[0])))
        #Creating the adjacency matrix and weight matrix
        for i in range(X.shape[0]):
            for j in range(arr.shape[1]):
                wtmatrix[i,arr[i,j]] = dist[i,j]        
        #evaluating the local measures
        measure = 0
        sum=0
        Y_transformed = np.zeros((X.shape[0],X.shape[1]))
        for i in range(X.shape[0]):
            Xtemp = np.zeros((nbor,X.shape[1]))
            Ytemp = np.zeros((nbor,Y.shape[1]))
            for j in range(nbor):    
                m = arr[i,j]
                Xtemp[j,:] = X[m,:]
                Ytemp[j,:] = Y[m,:]
            Xtemp = Xtemp.T
            Ytemp = Ytemp.T
            value,P,D,t, cnt = asim(Xtemp,Ytemp,nbor)
            measure = measure + value
            sum=sum+cnt
            Ytemp = np.zeros((Y.shape[1],1))
            Ytemp[:,0] = Y[i,:]
            Y_transformed[i,:] = (np.dot(np.dot(P,D),Ytemp) +t).T
        local_measure = (1/X.shape[0])*measure
        count = (1/X.shape[0])*sum
        
        return local_measure,Y_transformed, count
    
    def localProcrustes(X,Y):
        def procrustes(X_norm,Y_norm):
            if(X_norm.shape[1]>Y_norm.shape[1]):
                Y_norm = np.concatenate((Y_norm, np.zeros((X_norm.shape[0], X_norm.shape[1]-Y_norm.shape[1]))), 1)
            A= np.dot((X_norm.T),Y_norm);
            U,S,V=np.linalg.svd(A,full_matrices = False);
            d= 1-(S.sum())**2;
            return d
        measure1 = 0;
        X_temp = np.zeros((nbor,X.shape[1]))
        Y_temp = np.zeros((nbor,Y.shape[1]))
        knn = NearestNeighbors(n_neighbors=nbor) 
        knn.fit(Y)
        arr = knn.kneighbors(return_distance=False)
        count1 = 0
        #print(arr)
        for i in range(X.shape[0]):
            for b in range(nbor):
                X_temp[b,:]=X[arr[i,b],:]
                Y_temp[b,:]=Y[arr[i,b],:]
            mu_X=X_temp.mean(axis=0);
            mu_Y=Y_temp.mean(axis=0);
            X_norm=(X_temp-mu_X);
            Y_norm=(Y_temp-mu_Y);
            ss_x = (X_norm ** 2.).sum()
            ss_y = (Y_norm**2.).sum()
            norm_x = np.sqrt(ss_x)
            norm_y = np.sqrt(ss_y)
            if(norm_y == 0):
                count1 = count1 + 1
                continue
            X_norm = X_norm/norm_x;
            Y_norm = Y_norm/norm_y;
            measure1= measure1 + procrustes(X_norm,Y_norm) 
        measure1 = measure1/(X.shape[0]-count1)    
        return measure1
    
    def KSmeasure(X,Y):
        Xtemp = X
        Ytemp = Y
        for i in range(X.shape[1]):
            Xtemp[:,i] = np.sort(X[:,i])
            Ytemp[:,i] = np.sort(Y[:,i])
        def F_n(X,a):
            if (a < X[0]):
                return 0
            if(a > X[X.shape[0]-1]):
                return 1
            i = 0
            Fx = 0
            while((a > X[i]) and (i< X.shape[0])):
                Fx = Fx + 1/(X.shape[0])
                i = i+1
            return Fx
        
        Combined = np.zeros((2*X.shape[0],X.shape[1]))
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Combined[i,j] = X[i,j]
                Combined[i+X.shape[0],j] = Y[i,j]
                
        Measures = np.zeros(X.shape[1])
        for i in range(2*X.shape[0]):
            for j in range(X.shape[1]):
                if(Measures[j]< abs(F_n(Xtemp[:,j],Combined[i,j]) - F_n(Ytemp[:,j],Combined[i,j]))):
                    Measures[j] = abs(F_n(Xtemp[:,j],Combined[i,j]) - F_n(Ytemp[:,j],Combined[i,j]))
                    
        final_measure = Measures.sum()/Measures.shape[0]

        return final_measure
    
    def procrustes2(x=np.array([]), y=np.array([]), scaling=True, reflection=True):
        n, m = x.shape
        ny, my = y.shape
        mu_x = x.mean(0)
        mu_y = y.mean(0)
        x0 = x - mu_x
        y0 = y - mu_y
        ss_x = (x0 ** 2.).sum()
        ss_y = (y0**2.).sum()
        norm_x = np.sqrt(ss_x)
        norm_y = np.sqrt(ss_y)
        x0 /= norm_x
        y0 /= norm_y
        if my < m:
            y0 = np.concatenate((y0, np.zeros((n, m - my))), 1)
        a = np.dot(x0.T, y0)
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        v = vt.T
        t = np.dot(v, u.T)
        if reflection is not True:
            have_reflection = np.linalg.det(t) < 0
            if reflection != have_reflection:
                v[:, -1] *= -1
                s[-1] *= -1
                t = np.dot(v, u.T)
        trace_ta = s.sum()
        if scaling:
            b = trace_ta * norm_x / norm_y
            d = 1 - trace_ta**2
            z = norm_x*trace_ta*np.dot(y0, t) + mu_x
        else:
            b = 1
            d = 1 + ss_y/ss_x - 2 * trace_ta * norm_y / norm_x
            z = norm_y*np.dot(y0, t) + mu_x
        if my < m:
            t = t[:my, :]
        c = mu_x - b*np.dot(mu_y, t)
        transform = {'rotation': t, 'scale': b, 'translation': c}
        return d
    A,Y_transformed, count = nieqa(X.T,Y.T)
    B = localProcrustes(X,Y_transformed)   
    C = 0
    for i in range(X.shape[1]):
        d1 = X[:,i]
        d2 = Y_transformed[:,i]
        stat,p = (ks_2samp(d1,d2))
        C = C + stat/X.shape[1]
    return A,B, Y_transformed, count

# For Visualising data 
def visualise(X):
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen','white']
    maxi=np.amax(X[:,0])
    mini=np.amin(X[:,0])
    rangei=maxi-mini

    data_colour=np.zeros(len(X),int)

    for i in range (len(X)):
        data_colour[i]=math.floor((X[i][0]-mini)/(rangei/8))
    fig = plt.figure(figsize=(35, 25))    
    ax = fig.add_subplot(444, projection='3d')
    for i in range (len(X)):
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=colors[data_colour[i]])
    #plt.title('Histogram of Arrival Delays')
    plt.xlabel('X')
    plt.ylabel('Y') 
    plt.show()
    
    
# **********************Reading dataset*************************

df = pd.read_csv('S_Curve.csv')
X = np.array(df)

embedding = Isomap(n_components=2) # Isomap Embedding
# embedding = LocallyLinearEmbedding(n_components=2) # LLE Embedding
# embedding = TSNE(n_components=2) # TSNE Embedding
# embedding = MDS(n_components=2) # MDS Embedding
# embedding = umap.UMAP(n_components=2) # umap Embedding

F = embedding.fit_transform(X)

# *************For auto encoder file uncomment below code****************
# df = pd.read_csv('auto_scurve.csv')
# F = np.array(df)

# **************Calculation***************
start=time.time()
a,b,Y_Transformed, count = localplusglobal(X,F)
end=time.time()

mse = mean_squared_error(Y_Transformed, X)
toe = end - start

print("L1 Measure: ", a)
print("Local Proscrustes Measure: ", b)
print("time taken: ", toe)
print("Number of iteration: ", count)
print("Mean Squared Error: ", mse)

# visualising the transformed Y
visualise(Y_Transformed)