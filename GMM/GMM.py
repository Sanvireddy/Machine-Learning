# -*- coding: utf-8 -*-




#importing the required modules
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import operator

print('************** Part - A **************')
print('')
print('Question-1')

#reading the csv file
data=pd.read_csv(r'C:\Users\Dell\Desktop\seismic_bumps1.csv')

#deleting all the columns of data in attr
attr=['nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89']
for i in range(len(attr)):
    del data[attr[i]]

#obtaining the data classs wise
df=data.groupby('class')

#obtaining data with respect to class 0 and 1
df1=df.get_group(0)
df2=df.get_group(1)

#splitting the data into 70% train data and 30% test data of each class
X_label1=df1.iloc[:,-1].values
X_label2=df2.iloc[:,-1].values

X1=df1.iloc[:,0:-1].values
X2=df2.iloc[:,0:-1].values

X_train1,X_test1,X_label_train1,X_label_test1=train_test_split(X1, X_label1, test_size=0.3, random_state=42)
X_train2,X_test2,X_label_train2,X_label_test2=train_test_split(X2, X_label2, test_size=0.3, random_state=42)

#concatenating label train data of each class
#concatenating label test data of each class
c=[X_label_train1,X_label_train2]
d=[X_label_test1,X_label_test2]
X_label_train=np.concatenate(c)
X_label_test=np.concatenate(d)

#calculating length of train and test label
a=len(X_train1)
b=len(X_train2)
#creating a new list of predicted data from the train data 
#which is fitted in gaussian mixture model with q=2,4,8,16 components
#evaluating log likelihood values of test data of each class
#predicting whether it belongs to clxtraiass zero or one
#appending predicted class in xpred for each q
#finding confusion matrix and accuracy score using in-built functions
#we can see that it matched the results of previous lab for q=1

ca=[]
for i in range(1,5):
    xpred=[]
    
    gmm1 = GaussianMixture(n_components=2**i ,covariance_type='full',random_state=42).fit(X_train1)
    gmm2 = GaussianMixture(n_components=2**i, covariance_type='full',random_state=42).fit(X_train2)
    
    l1=gmm1.score_samples(X_test1)
    l2=gmm2.score_samples(X_test1)
    for j in range(len(X_test1)):
        if (((math.exp(l1[j]))*a)/(a+b))>(((math.exp(l2[j]))*b)/(a+b)):
            xpred.append(0)
        else:
            xpred.append(1)  
            
    l3=gmm1.score_samples(X_test2)
    l4=gmm2.score_samples(X_test2)
    for k in range(len(X_test2)):
        if (((math.exp(l3[k]))*a)/(a+b))>(((math.exp(l4[k]))*b)/(a+b)):
            xpred.append(0)
        else:
            xpred.append(1)
            
    cm=confusion_matrix(X_label_test,np.array(xpred))
    print('(a) Confusion matrix for Q = ',2**i,' is ')
    print(cm)
    print()
    print('(b) Classification accuracy for Q =',2**i,'is ',round(accuracy_score(X_label_test,xpred)*100,3),'%')
    print('')
    ca.append(round(accuracy_score(X_label_test,xpred)*100,3))
dict={'KNN(before norm)':[93.1701],'KNN(after norm)':[92.9124],'Bayes Classifier unimodal guassian density':[87.5],'Bayes classifier using GMM':[max(ca)]}
p=pd.DataFrame(dict)
print(p)

#*****************************************************
print('************** Part - B **************')
print('')
print('Question-1(a)')
#reading the csv file
data=pd.read_csv(r'C:\Users\Dell\Desktop\atmosphere_data.csv')

#splitting the data to test data and train data each of size 30% and 70%
X_train,X_test=train_test_split(data, test_size=0.3, random_state=42)
at=['humidity','pressure','rain','lightAvg','lightMax','moisture','temperature']

#converting the train and test arrays into data frame
xtrain=pd.DataFrame(data=X_train,columns=at)
xtest=pd.DataFrame(data=X_test,columns=at)

#converting dataframes to csv file
xtrain.to_csv("atmosphere-train.csv")
xtest.to_csv("atmosphere-test.csv")

#obtaining pressure and temperature train and test values as an array
X_train=(xtrain['pressure']).to_numpy().reshape(-1,1)
Y_train=(xtrain['temperature']).to_numpy().reshape(-1,1)
X_test=xtest['pressure'].to_numpy().reshape(-1,1)
Y_test=xtest['temperature'].to_numpy().reshape(-1,1)

#obtaining the linear regression inbuilt funtion and fitting xtrain and ytrain
#predicting train and test data after fitting train data into linear regression
regression=LinearRegression()
regression.fit(X_train,Y_train)
Y_pred = regression.predict(X_test)
Y_predi=regression.predict(X_train) 
#print(Y_pred)

#plotting the best fit line on the training data where x-axis is pressure value and y-axis is temperature.
plt.scatter(X_train, Y_train, color ='b') 
plt.plot(X_train, Y_predi, color ='r') 
plt.xlabel('Train data of pressure') 
plt.ylabel('Train data of pressure')
plt.title('Plot of best fit line on training data')
plt.show()

 
#*****************************************
print('')
print('Question-1(b)')
#obtaining the mean squared error of train and predicted temperature
e1=mean_squared_error(Y_train,Y_predi)
print("Prediction accuracy on the training data using root mean squared error is ",round(e1**0.5,3))

#*******************************************
print('Question-1(c)')
print('')
#obtaining the mean squared error of test and predicted temperature
e2=mean_squared_error(Y_test,Y_pred)
print("Prediction accuracy on the test data using root mean squared error is ",round(e2**0.5,3))

#plotting the scatter plot of actual temperature (x- axis) vs predicted temperature (y-axis) onthe test data
plt.scatter(Y_test,Y_pred)
plt.xlabel('Actual temperature')
plt.ylabel('Predicted temperature')
plt.title('Scatter plot of actual and predicted temperature')
plt.show()


#*****************************************************
print('Question-2')
print('')

#fitting non linear regression model using polynomial curve fitting  using inbuilt function
#creating two empty lists of rmse values
#appending rmse values of each value of p of train data to rmse1
#appending rmse values of each value of p test data to rmse2
rmse1=[]
rmse2=[]
for p in range(2,6):
    poly = PolynomialFeatures(degree=p)
    x_poly = poly.fit_transform(X_train)
    poly.fit(x_poly,Y_train)
    lin = LinearRegression()
    lin.fit(x_poly,Y_train)
    
    y_pred = lin.predict(poly.fit_transform(X_test))
    y_predi = lin.predict(poly.fit_transform(X_train))
    
    rmse1.append(np.sqrt(mean_squared_error(Y_train,y_predi)))
    rmse2.append(np.sqrt(mean_squared_error(Y_test,y_pred)))
    print('(a) Prediction accuracy on the train data for p = ',p,' is ',round(np.sqrt(mean_squared_error(Y_train,y_predi)),3))
    print("(b) Prediction accuracy on the test data for p =",p,' is ',round(np.sqrt(mean_squared_error(Y_test,y_pred)),3))
    print('')

#plot of prediction accuracy of training data 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X=[2,3,4,5]
ax.bar(X,rmse1)
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE')
plt.title('Prediction accuracy of training data')
plt.show()

#plot of prediction accuracy of testing data
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X=[2,3,4,5]
ax.bar(X,rmse2)
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE')
plt.title('Prediction accuracy of testing data')
plt.show()

#**************************************************
print('Question-2(c)')

#Plotting the best fit curve using best fit model on the training data where x-axis is pressure value and y-axis is temperature.
x=X_train
y=y_predi
sort_axis=operator.itemgetter(0)
sorted_zip=sorted(zip(x,y),key=sort_axis)
x,y=zip(*sorted_zip)
plt.plot(x,y,color='r',linewidth=2)

plt.scatter(X_train,Y_train)
plt.title('Plot of best fit curve ')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.show()

#*****************************************************

print('Question-2(d)')
#Plotting the scatter plot of actual temperature (x-axis) vs predicted temperature (y-axis) on the test data for the best degree of polynomial (p)
plt.scatter(Y_test,y_pred)
plt.title('Scatter plot of actual vs predicted temperature')
plt.xlabel('Actual temperature')
plt.ylabel('Predicted temperature')
plt.show()










