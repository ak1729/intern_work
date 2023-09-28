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

nbor=5

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



# ************************Gaussian Process***********************

def objective(params):
    alpha = params[0]
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, F, X, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    
    return mse
start=time.time()
space = [(1e-40, 1e40)]
res = gp_minimize(objective, space, n_calls=100, random_state=0)
print(f'Best hyperparameters: {res.x}')

alpha = res.x[0]
model = Ridge(alpha=alpha)
model.fit(F, X)
predicted_X = model.predict(F)
end=time.time()
tot=end-start
localProGauss = localProcrustes(X, predicted_X)
mse = mean_squared_error(predicted_X, X)

print("Local Proscrustes Measure: ", localProGauss)
print("time taken: ", tot)
print("Mean Squared Error: ", mse)

# For visualisation
visualise(predicted_X)