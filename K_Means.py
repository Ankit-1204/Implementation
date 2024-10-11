#K-Mean Clustering
import pandas as pd
import numpy as np

class cluster:
    def __init__(self,k):
        self.k=k
    
    def _predict(self,x):
        random_indices = np.random.choice(x.shape[0], size=self.k, replace=False)
        centroids=x[random_indices]
        final_cent=[]
        final_centroid=[]
        for k in range(100):
            cent=[]
            for point in x:
                distances=np.linalg.norm(point-centroids,axis=1)
                center=np.argmin(distances)
                cent.append(center)
                
            for i in range(self.k):
                indices = np.where(cent == i)[0]
                if len(indices)>0:
                    centroids[i]=np.mean(x[indices],axis=0)
                else:
                    r=np.random.choice(x.shape[0],size=1,replace=False)
                    centroids[i]=x[r]
            final_cent=cent
            final_centroid=centroids
        return final_cent , final_centroid

