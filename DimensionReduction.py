"""
Created on Fri Oct 16 13:42:16 2020

@author: sanvireddy
"""



#importing required modules

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

#reading csv file
data=pd.read_csv(r'landslide_data3.csv')
#creating a list of attribute nammes
attribute=['dates','stationid','temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']


#calculating quartile 1 and quartile 2
#evaluating lower whisker and higher whisker
#replacing the outliers of the data with median
for x in range(2,len(attribute)):
    Q1=data[attribute[x]].quantile(0.25)
    Q3=data[attribute[x]].quantile(0.75)
    IQR=Q3-Q1
    lower_whisker=Q1-(1.5*IQR)
    upper_whisker=Q3+(1.5*IQR)
    outlier=[]
    y=data[attribute[x]].median()
    for a in data[attribute[x]]:
        if a<lower_whisker or a>upper_whisker:
             outlier.append(a)
    for i in range(len(outlier)):
        data[attribute[x]].replace({outlier[i]:y},inplace=True)
#copying df
data1=data.copy() 

#***************** 

print('Question-1(a)')  
#calculating minimum and maximum of each attribute
#min-max normalisation of each attribute given minimum and maximum
for x in range(2,len(attribute)):
    a=data1[attribute[x]].min()
    b=data1[attribute[x]].max()
    data1[attribute[x]]=((((data1[attribute[x]]-a)*6)/(b-a))+3)

#creating dictionary which is converted to dataframe showing minimum and maximum of both new and old data
dict1={'Statistic':['Minimum','Maximum','New-Mininum','New-Maximum']}
for x in range(2,len(attribute)):
    dict1[attribute[x]]=[data[attribute[x]].min(),data[attribute[x]].max(),data1[attribute[x]].min(),data1[attribute[x]].max()]
df1=pd.DataFrame.from_dict(dict1)
df1=df1.transpose() 
print(df1)

#**********
print(' ')
print('Question-1(b)')
data2=data1.copy()
#copying data

#calculating mean and standard deviation
#standardizing each selected attribute using mean and standard deviation
for x in range(2,len(attribute)):
    a=data2[attribute[x]].mean()
    b=data2[attribute[x]].std()
    data2[attribute[x]]=((data2[attribute[x]]-a)/b)

#creating dictionary which is converted to dataframe showing mean and standard deviation of both new and old data
dict2={'Statistic':['Mean','Standard Deviation','New- Mean','New- Standard Deviation']}
for x in range(2,len(attribute)):
    dict2[attribute[x]]=[data1[attribute[x]].mean(),data[attribute[x]].std(),data2[attribute[x]].mean(),data2[attribute[x]].std()]
df2=pd.DataFrame.from_dict(dict2)
df2=df2.transpose() 
print(df2) 

#****************************************************
print('Question-2a')

#creating 2 dimensional synthetic data of 1000 samples which are independently and identically distributed with bi-variate Gaussian distribution
n=1000
mean=[0,0]
cov=[[6.84806467,7.63444163],[7.63444163,13.02074623]]
data3=np.random.multivariate_normal(mean,cov,n)

#creating a covariance matrix
#plotting scatter plot of the data
m=np.array(cov)
plt.scatter(data3[:,0],data3[:,1],marker='+')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Plot of 2D synthetic data')
plt.show()
#*******************************************************
print('Question-2b')

#creating origin
#evaluating eigen values and eigen vectors of covariance

origin=[0,0]
eig_values,eig_vectors=np.linalg.eig(m)

#taking each eigen value separately into a list
eig_vec1=eig_vectors[:,0]
eig_vec2=eig_vectors[:,1]

#obtaining the scatter plot of the data
#obtaining the direction of each eigen vector
plt.scatter(data3[:,0],data3[:,1],marker='+')
plt.quiver(*origin,*eig_vec1,scale=5)
plt.quiver(*origin,*eig_vec2,scale=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Plot of 2D synthetic data and eigen directions')
plt.axis('equal')
plt.show()
#***********
print('Question-2c')

#obtaining the scatter plot of the data
#obtaining the direction of each eigen vector
#projecting data onto each eigen vector
#obtain plot of projected data

plt.scatter(data3[:,0],data3[:,1],marker='+')
plt.quiver(*origin,*eig_vec1,scale=5)
plt.quiver(*origin,*eig_vec2,scale=5)
plt.axis('equal')
plt.title('Projected values onto first eigen vector')

vec1=np.array(eig_vec1)
for j in range(1000):
    v1=np.array([data3[j][0],data3[j][1]])
    p1=(v1.dot(vec1)/vec1.dot(vec1))*vec1
    plt.scatter(p1[0],p1[1],color='r',marker='+')
plt.show()

plt.scatter(data3[:,0],data3[:,1],marker='+')
plt.quiver(*origin,*eig_vec1,scale=5)
plt.quiver(*origin,*eig_vec2,scale=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Projected values onto the second eigen directions')
plt.axis('equal')
vec2=np.array(eig_vec2)

for j in range(1000):
    v2=np.array([data3[j][0],data3[j][1]])
    p2=(v2.dot(vec2)/vec2.dot(vec2))*vec2
    plt.scatter(p2[0],p2[1],color='r',marker='+')
plt.show()

#projecting data on to eigen vectors
projd=np.dot(data3,eig_vectors)

#******************
print('Question-2(d)')

#reconstructing data using eigen vectors
#caluclating reconstruction error using mean square error
data3_rec=np.dot(projd,eig_vectors.T)   
print('Reconstruction error= ',((data3-data3_rec)**2).sum()/len(data3))

#********************************************8
print('Question-3a')

#dropping dates and station id from original data
data2.drop(['dates','stationid'],axis=1,inplace=True)

#Performing principle component analysis (PCA) on outlier corrected standardized data
#Reducing the multidimensional data into lower dimensions
#finding variance of the projected data along the two directions
#calculating eigenvalues of the two directions of projection

pca=PCA(n_components=2)
Data=pca.fit_transform(data2)
pca_data=pd.DataFrame(data=Data,columns=['pca1','pca2'])
covar=pca_data.cov()
eigval,eigvec=np.linalg.eig(covar)#Eigen Value and Eigen Vector
print(eigval)

#evaluating covariance matrix

covar1=data2.cov()
eigenva,eigenve=np.linalg.eig(covar1)
eigenva=list(eigenva)
eigenva.sort(reverse=True)


for i in range(2):
    print('Variance along Eigen Vector',i+1,':',np.var(Data.T[i]))
    print('Eigen Value corresponding to Eigen Vector',i+1,':',eigval[i])
    print('')

#obtaining scatter plot of the reduced dimensional data
plt.scatter(Data[:,0],Data[:,1],marker='+')
plt.xlabel('Principal Component-1')
plt.ylabel('Principal Component-2')
plt.title('Scatter plot of reduced dimensional data')
plt.show()

#**********************************************
print('Question-3b')

#plotting eigen values of 
X=[0,1,2,3,4,5,6]
Y=eigenva
plt.plot(X,Y)
plt.ylabel('Eigen values')
plt.title('Plot of eigen values')
plt.show()

#*********************************

print('Question-3c')

#reconstruction errors in terms of RMSE considering the different values of l 
#performing back projection

RMSE=[]
for i in range(1,8):
    pca=PCA(n_components=i)
    Data=pca.fit_transform(data2)
    inv_trans=pca.inverse_transform(Data)
    RMSE.append((((data2.values-inv_trans)**2).mean()**.5))
X=[1,2,3,4,5,6,7]
Y=RMSE
plt.plot(X,Y)
plt.title('Plot of reconstruction error')
plt.xlabel('l')
plt.ylabel('RMSE values')
plt.show()
