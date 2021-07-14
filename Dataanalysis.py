# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:10:02 2020

@author: Sanvi Reddy
"""
#Name : Sanvi Reddy
#Registration number : B19193
#Contact number : 7675851126

#importing the required modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
#reading csv file
data=pd.read_csv(r'C:\Users\Sanvi Reddy\Downloads\landslide_data3.csv').fillna(value=0)
dl=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
print("Question-1")
#creating empty lists of mean, mode, median, minimum, maximum, standard deviaton and storing the respective values in it
Mean=[]
Median=[]
Mode=[]
Minimum=[]
Maximum=[]
Standard_Deviation=[]
for i in range(len(dl)):
#appending value of mean of attribute dl[i]
    Mean.append(data[dl[i]].mean())
    print("Mean of " + dl[i] + " is " + str(Mean[i]))
#appending the value of median of attribute dl[i]
    Median.append(data[dl[i]].median())
    print("Median of " + dl[i] + " is " + str(Median[i]))
#appending the value of mode of attribute dl[i]
    Mode.append(data[dl[i]].mode()[0])
    print("Mode of " + dl[i] + " is " + str(Mode[i]))
#appending the value of minimum of attribute dl[i]
    Minimum.append(data[dl[i]].min())
    print("Minimum of " + dl[i] + " is " + str(Minimum[i]))
#appending the value of maximum of attribute dl[i]
    Maximum.append(data[dl[i]].max())
    print("Maximum of " + dl[i] + " is " + str(Maximum[i]))
#appending the value of standard deviation of attribute dl[i]
    Standard_Deviation.append(data[dl[i]].std())
    print("Standard Deviation of " + dl[i] + " is " + str(Standard_Deviation[i]))
    print('   ')
#************************************************
print("Question-2(a)")
#creating list of attributes which we have to plot against rain
dq=['temperature','humidity','pressure','lightavgw/o0','lightmax','moisture']

for j in range(len(dq)):
#using in-built functions and plotting scatter plot of rain vs other attributes 
    x=data.rain
#taking rain on x-axis
    y=data[dq[j]]
#taking attribute dq[j] on y-axis
    plt.scatter(x,y)
#obtaining scatter plot 
    plt.title("Scatter plot between rain and "+ dq[j])
    plt.xlabel('X-Axis rain')
    plt.ylabel('Y-Axis '+ dq[j])
    plt.show()  
    
#************************************************    
print("Question-2(b)") 
#creating list of attributes which we have to plot against temperature   
dp=['humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
for k in range(len(dp)):
#using in-built functions and plotting scatter plot of temperature vs other attributes
#taking rain on x-axis
#obtaining scatter plot 
    x=data.temperature
    y=data[dp[k]]
    plt.scatter(x,y)
    plt.title("Scatter plot between temperature and " + dp[k])
    plt.xlabel('X-Axis temperature')
    plt.ylabel('Y-Axis ' + dp[k])
    plt.show()    


print(' ')    
#************************************************   


print("Question-3(a)")
for j in range(len(dq)):
    x=data.rain
    y=data[dq[j]]
#finding standard correlation coefficient which is also known as pearson correlation coefficient
    corr=np.corrcoef(x,y)
    print("Correlation coefficient of rain and "+dq[j],"     :  %.3f"%corr[0,1])
print("\n")
for k in range(len(dp)):
    x=data.temperature
    y=data[dp[k]]
    corr=np.corrcoef(x,y)
    print("Correlation coefficient of temperature and "+dp[k],"     :  %.3f"%corr[0,1])
print("\n")

#************************************************

#obtaining a histogram using in-built function 
print("Question-4")
#histogram of attribute rain
p=data['rain']
p.hist()
plt.title("Histogram of rain")
plt.xlabel('X-axis - rain')
plt.ylabel('Y-axis - frequency')
plt.show()

#histogram of attribute moisture
q=data['moisture']
q.hist()
plt.title("Histogram of moisture")
plt.xlabel('X-Axis - moisture')
plt.ylabel('Y-Axis - frequency')
plt.show()
print(' ')


#**********************************************************")
print("Question-5")
y=data['rain']
x=data['stationid']
df = DataFrame({'id':x, 'reading':y})
grouped = df.groupby('id')

for group in grouped:
    plt.figure()
    plt.hist(group[1].reading)
    plt.xlabel('rain')
    plt.ylabel('frequency')
    plt.title('Histogram of rain in stationid ' + group[0])
    plt.show()
print(' ')

#*********************************************************************
print("Question-6")
data.boxplot(column='rain',grid=False,whis=[15,99])
plt.ylabel('rain(in mm)')
plt.title('Boxplot for rain')
plt.show()

data.boxplot(column='moisture',grid=False)
plt.ylabel('moisture')
plt.title('Boxplot for moisture')
plt.show()
