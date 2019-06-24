import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster.k_means_ as kmean

'''
K means: most popular cluster algorithm
Cluster the unsupervised data into K clusters
Select K is important
'''
file = pd.read_csv('wine.data')
data = file.values
y_data = data[:, 0]
x_data = data[:, 1:]

shuffle = np.random.permutation(x_data.shape[0])
x_data = x_data[shuffle]
y_data = y_data[shuffle]

total = x_data.shape[0]
train_end = int(total * 0.8)
x_train = x_data[0:train_end, :]
y_train = y_data[0:train_end]
x_test = x_data[train_end:, :]
y_test = y_data[train_end:]  

def PCA(x, k = 2):
    cov_x = np.cov(x.T)
    u, s, v = np.linalg.svd(cov_x)
    project_m = u[:, 0:k]
    pca_x = np.matmul(x, project_m)
    
    return pca_x

def Normalize(x):
    '''
    Normalization: to 2D data feature
    '''
    num_feature = x.shape[1]
    
    for i in range(num_feature):
        Max = np.max(x[:, i])
        Min = np.min(x[:, i])
        x[:, i] = (x[:, i] - Min)/(Max - Min)
        
    return x

x_train = Normalize(x_train)
x_test = Normalize(x_test)


x_train = PCA(x_train)
x_test = PCA(x_test)

# K mean algorithm
k = 3
kmeans = kmean.KMeans( 
               n_clusters = k,
               n_init = 10, # number of time to select new cluster centroid of k means, 
               max_iter = 50 # number of iterations to run k means algorithm  
              ).fit(x_train)

print(x_train)

k1 = []
k2 = []
k3 = []
for i in range(x_train.shape[0]):
    if kmeans.labels_[i] == 0:
        k1.append(x_train[i])
    elif kmeans.labels_[i] == 1:
        k2.append(x_train[i])
    elif kmeans.labels_[i] == 2:
        k3.append(x_train[i])

k1 = np.array(k1)
k2 = np.array(k2)
k3 = np.array(k3)

plt.figure(1)
plt.scatter(x_train[:,0], x_train[:, 1])

plt.figure(2)
plt.scatter(k1[:, 0], k1[:, 1], label = 'cluster 1')
plt.scatter(k2[:, 0], k2[:, 1], label = 'cluster 2')
plt.scatter(k3[:, 0], k3[:, 1], label = 'cluster 3')
plt.legend()

plt.plot()







