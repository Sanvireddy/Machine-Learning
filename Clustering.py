# -*- coding: utf-8 -*-

#Name: Sanvi Reddy
#Registration number: B19193
#Contact number: 7675851126 

#importing the required modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy import spatial as spatial

#reading the csv file
train_data=pd.read_csv(r'C:\Users\Dell\Desktop\mnist-tsne-train.csv')
test_data=pd.read_csv(r'C:\Users\Dell\Desktop\mnist-tsne-test.csv')

#taking values in different list
l1=train_data['labels'].values
l2=test_data['labels'].values

#deleting the class information 
del train_data['labels']
del test_data['labels']
train=train_data.values
test=test_data.values

#fitting training data in kmeans model
#predicting class of train data
#with k value 10
K = 10
kmeans = KMeans(n_clusters=K,random_state=42)
kmeans.fit(train_data)
kmeans_prediction = kmeans.predict(train_data)

#**********************************************
print('Question-1(a)')

#obtaining scatter plot of data points
#after assigning them to different clusters
#plotting them with different colours
#obtaining centre of each cluster
#and plotting them
plt.scatter(train[:,0], train[:,1],marker='*', c=kmeans_prediction, s=25, cmap='rainbow')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('K Means of train data')
plt.show()

#**************************************************
print('Question-1(b)')
print('')

#defining a function to evaluate purity score
def purity_score(y_true, y_pred):
    #compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    #Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
p1=purity_score(l1,kmeans_prediction)
print('Purity score of train data(KMeans) is',p1)
print('')

#**************************************************

print('Question-1(c)')

#assigning the test examples onto cluster 
#plotting the test data points with different colours
#marking the centers of each cluster
kmeans_test=kmeans.predict(test_data)

plt.scatter(test[:,0], test[:,1], c=kmeans_test, marker='*',s=50, cmap='rainbow')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=200, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('K Means of test data')
plt.show()

#******************************************************

#computing the purity score 
#after training examples are assigned to clusters
print('Question-1(d)')
print('')
t1=purity_score(l2,kmeans_test)
print('Purity score of test data(KMeans) is',t1)

#****************************************************

print('')
print('Question-2(a)')

#fitting the train data into gmm model
#with 10 clusters
K = 10
gmm = GaussianMixture(n_components = K,random_state=42)
gmm.fit(train_data)
GMM_prediction = gmm.predict(train_data)

#plotting the data points with different colours for each cluster
#marking the centres of the clusters in the plot
plt.scatter(train[:,0], train[:,1], c=GMM_prediction,marker='*', s=25, cmap='rainbow')

centers=gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('GMM of train data')
plt.show()

#******************************************************

print('')
print('Question-2(b)')

#computing the purity score after training examples are assigned to clusters
p2=purity_score(l1,GMM_prediction)
print('Purity score of train data(GMM) is',p2)

#******************************************************
print('')
print('Question-2(c)')

#assigning the test examples onto cluster
#plotting the test data points with different colours for each clusters
#Mark the centres of the clusters in the plot
gmm_test=gmm.predict(test_data)

plt.scatter(test[:,0], test[:,1], c=gmm_test,marker='*', s=25, cmap='rainbow')

centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('GMM of test data')
plt.show()

#*******************************************************
print('Question-2(d)')

#computing the purity score 
#after training examples are assigned to clusters
t2=purity_score(l2,gmm_test)
print('Purity score of test data(GMM) is',t2)

#****************************************************************

print('')
print('Question-3(a)')

#applying dbscan clustering with eps=5 and min_samples=10
#fitting train data into dbscan model
dbscan_model=DBSCAN(eps=5, min_samples=10).fit(train_data)
DBSCAN_predictions = dbscan_model.labels_

#plotting the data
plt.scatter(train[:,0], train[:,1], c=DBSCAN_predictions,marker='*', s=25, cmap='rainbow')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('DBSCAN of train data')
plt.show()

#*********************************************************

print('Question-3(b)')
#computing the purity score of train data
p3=purity_score(l1,DBSCAN_predictions)
print('Purity score of train data(DBSCAN) is',p3)

print('')

#***********************************************************

print('Question-3(c)')

#fitting the test data into the predicted model
#assigning the test examples onto cluster


def dbscan_predict(model, X):
    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new

dbtest=dbscan_predict(dbscan_model, test)

#plotting the test data points with different colours for each cluster
plt.scatter(test[:,0], test[:,1], c=dbtest,marker='*', s=25, cmap='rainbow')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('DBSCAN of test data')
plt.show()

#***********************************************

print('Question-3(d)')

#Computing the purity score 
#after training examples are assigned to clusters
t3=purity_score(l2,dbtest)
print('Purity score of test data(DBSCAN) is',t3)


#****************Bonus-Question*******************************
print('')
print('Bonus Question')
 
#taking different values of k
K = [2,5,8,12,18,20] 

#creating an empty list for distortion measure(K means) 
kdist = []

#appending distortion measure of all values of k in K
for k in K:
    kmeanModel = KMeans(n_clusters=k,random_state=42).fit(train_data)
    kmeanModel.fit(train_data)
    kmeans_prediction = kmeanModel.predict(train_data)
    print('Purity score of train data for k value is '+ str(k)+' is ',purity_score(l1,kmeans_prediction))
    ktest=kmeanModel.predict(test_data)
    print('Purity score of test data for k value is '+ str(k)+' is ',purity_score(l2,ktest))
    kdist.append(sum(np.min(cdist(train_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / train_data.shape[0])


# Plot the elbow
plt.plot(K, kdist, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k(K Means)')
plt.show()


#***************Plot of K means for k=8(BEST PLOT)*********************

#*******************for train data*******************************

kmeans = KMeans(n_clusters=8,random_state=42)
kmeans.fit(train_data)
kmeans_prediction = kmeans.predict(train_data)

plt.scatter(train[:,0], train[:,1],marker='*', c=kmeans_prediction, s=25, cmap='rainbow')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('K Means of train data')
plt.show()


#*******************for test data**************

kmeans_test=kmeans.predict(test_data)

plt.scatter(test[:,0], test[:,1], c=kmeans_test, marker='*',s=50, cmap='rainbow')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=200, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('K Means of test data')
plt.show()


#*************************************************

#creating an empty list of distortion measures(gmm)
gdist=[]
print('GMM')
#appending distortion measures of all values of k in K
for i in K:
    gmm_model=GaussianMixture(n_components = i,random_state=42)
    gmm_model.fit(train_data)
    gmmtrain=gmm_model.predict(train_data)
    print('Purity score of train data for k value '+ str(i)+' is ',purity_score(l1,gmmtrain))
    gmmtest=gmm_model.predict(test_data)
    print('Purity score of test data for k value '+ str(i)+' is ',purity_score(l2,gmmtest))
    gdist.append(sum(np.min(cdist(train_data, gmm_model.means_, 'euclidean'), axis=1)) / train_data.shape[0])


#plotting the elbow
plt.plot(K, gdist,'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k(GMM)')
plt.show()



#***********plot of GMM for k=8(BEST PLOT)************

#******************for train data**********************

gmm = GaussianMixture(n_components = 8,random_state=42)
gmm.fit(train_data)
GMM_prediction = gmm.predict(train_data)
plt.scatter(train[:,0], train[:,1], c=GMM_prediction,marker='*', s=25, cmap='rainbow')

centers=gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('GMM of train data')
plt.show()


#*****************for test data***********************

gmm_test=gmm.predict(test_data)

plt.scatter(test[:,0], test[:,1], c=gmm_test,marker='*', s=25, cmap='rainbow')

centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.2)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('GMM of test data')
plt.show()

#***************************dbscan********************************

#dbscan
p=[1,5,10]
for i in p:
    dbscan_model=DBSCAN(eps=i, min_samples=10).fit(train_data)
    DBSCAN_predictions = dbscan_model.labels_
    
    plt.scatter(train[:,0], train[:,1], c=DBSCAN_predictions,marker='*', s=25, cmap='rainbow')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('DBSCAN of train data(Eps='+str(i)+', Min_samples=10)')
    plt.show()

    contingency_matrix=metrics.cluster.contingency_matrix(l1,DBSCAN_predictions)#compute contingency matrix (also called confusion matrix)

    #Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix) # Return cluster accuracy
    
    print('Purity score, Min_Samples=10, Epsilon=',i)
    print('Train data: ',contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix))
    
    
    def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean): 
        # Result is noise by default 
        y_new = np.ones(shape=len(X_new), dtype=int)*-1
        # Iterate all input samples for a label 
        for j, x_new in enumerate(X_new): 
            # Find a core sample closer than EPS 
            for i, x_core in enumerate(dbscan_model.components_): 
                if metric(x_new, x_core) < dbscan_model.eps: 
                    # Assign label of x_core to x_new 
                    y_new[j] =dbscan_model.labels_[dbscan_model.core_sample_indices_[i]] 
                    break 
        return y_new 
    dbtest = dbscan_predict(dbscan_model,test, metric =spatial.distance.euclidean)

    plt.scatter(test[:,0], test[:,1], c=dbtest,marker='*', s=25, cmap='rainbow')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('DBSCAN of test data(Eps='+str(i)+', Min_samples=10)')
    plt.show()
    t3=purity_score(l2,dbtest)
    print('Test data:',t3)
    print('')


q=[1,10,30,50]
    
for i in q:
    dbscan_model=DBSCAN(eps=5, min_samples=i).fit(train_data)
    DBSCAN_predictions = dbscan_model.labels_
    
    plt.scatter(train[:,0], train[:,1], c=DBSCAN_predictions,marker='*', s=25, cmap='rainbow')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('DBSCAN of train data(Eps=5,Min_samples='+ str(i)+')')
    plt.show()

    contingency_matrix=metrics.cluster.contingency_matrix(l1,DBSCAN_predictions)#compute contingency matrix (also called confusion matrix)

    #Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix) # Return cluster accuracy
    
    print('Purity score, Eps=5 ,Min_samples=',i)
    print('Train data: ',contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix))
    
    
    def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean): 
        # Result is noise by default 
        y_new = np.ones(shape=len(X_new), dtype=int)*-1
        # Iterate all input samples for a label 
        for j, x_new in enumerate(X_new): 
            # Find a core sample closer than EPS 
            for i, x_core in enumerate(dbscan_model.components_): 
                if metric(x_new, x_core) < dbscan_model.eps: 
                    # Assign label of x_core to x_new 
                    y_new[j] =dbscan_model.labels_[dbscan_model.core_sample_indices_[i]] 
                    break 
        return y_new 
    dbtest = dbscan_predict(dbscan_model,test, metric =spatial.distance.euclidean)

    plt.scatter(test[:,0], test[:,1], c=dbtest,marker='*', s=25, cmap='rainbow')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('DBSCAN of test data(Eps=5,Min_samples='+ str(i)+')')
    plt.show()
    t3=purity_score(l2,dbtest)
    print('Test data:',t3)
    print('')
    













