import pandas as pd
from sklearn import cluster, datasets
from sklearn.cluster import KMeans
import numpy as np
from numpy import savetxt

#Reading train data set
train = pd.read_csv('train.csv')
#Reading test data set
test = pd.read_csv('test.csv')

#Kmean
print 'Making K-mean clustering'
k_means = cluster.KMeans(n_clusters=3)
n_digits = len(np.unique(train['label']))
k_means.fit(test)
KMeans(init='k-means++', n_clusters=10, n_init=n_digits,copy_x= True)
preds = [[index + 1,x] for index,x in enumerate( k_means.labels_)]
print preds
savetxt('submission2.csv', preds, delimiter=',',fmt='%d,%d',
            header='ImageId,Label', comments = '')
print 'File created!'
