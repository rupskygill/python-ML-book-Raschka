# Generate random data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                   n_features=2,
                   centers=3,
                   cluster_std=0.5,
                   shuffle=True,
                   random_state=0)

# plot data
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1], c='white', marker='o', s=50)
plt.grid()
plt.show()



# Use scikit Kmeans for clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
             init='random',   # use init='kmeans++' to initialize centroids using kmeans++
             n_init=10,
             max_iter=300,
             tol=1e-04,
             random_state=0)

# cluster
y_km = km.fit_predict(X)
# print SSE
print('Distortion: %.2f' % km.inertia_)

# Plot clustering
plt.scatter(X[y_km==0,0],X[y_km==0,1],s=50,c='lightgreen',marker='s',label='cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],s=50,c='orange',marker='o',label='cluster 2')
plt.scatter(X[y_km==2,0],X[y_km==2,1],s=50,c='lightblue',marker='v',label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=250,marker='*',c='red',label='centroids')
plt.legend()
plt.grid()
plt.show()

# Use elbow method to decide on value of k (no of clusters)
distortions = []
for i in range(1, 11):
     km = KMeans(n_clusters=i,
                 init='k-means++',
                 n_init=10,
                 max_iter=300,
                 random_state=0)
     km.fit(X)
     distortions.append(km.inertia_)
# plot
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

