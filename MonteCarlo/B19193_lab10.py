#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo integration: a method used to approximate a definite integral using random point generation

"""


# Importing required modules
import numpy as np

# Function to check whether a point lies inside a curve or not
def isUnderCurve(fun, x, y):
    if(fun(x, y) <= 0):
        return 1
    return 0


# Function to predict the required definite integral
# using Monte Carlo Method
def approxIntegral(func, size, show, num, a, b, c):
    print('('+ num + ')')
    for i in size:
        # Randomly generated n points
        XCord = (b - a)*np.random.random(i) + a
        YCord = c*np.random.random(i)
        
        # Rounding the points upto 3rd decimal place (Given)
        XCord = [round(i, 3) for i in XCord]
        YCord = [round(i, 3) for i in YCord]
        
        # List containing check of n elements;
        # If ith point lies inside the curve, then isPresent[i] = 1
        # Else it's value is 0
        isPresent = [isUnderCurve(func, x, y) for x,y in zip(XCord, YCord)]
        
        # Applying Monte Carlo Formula to above list isPresent, given a, b
        # and c, and Printing the result
        print('  '+ show +' \u2248 '+ str(c*(b - a)*np.mean(isPresent)) +' (n = '+ str(i) +')')
    
# Size list of no. of points to be generated
sampleSize = [100, 1000, 10000]

# ******* Question 1 *************

# Function to return value for a unit circle's (center{1,1}) equation S = 0
# where S1 = S(x, y)
def unitCircle(x, y):
    return ((x-1)*2 + (y-1)*2 - 1)

# Here, a = 0, b = 2, c = 2
approxIntegral(unitCircle, sampleSize, '\u03C0', '1', 0, 2, 2)

# ******* Question 2 *************

# Function to return value of given function y -f(x) = 0 for point (x1, y1)
def question2Func(x, y):
    return (y - (2/(1 + x**2)))

# Here, a = 0, b = 1, c = 2
approxIntegral(question2Func, sampleSize, '∫f(x)dx', '2', 0, 1, 2)

# ******* Question 3 *************

print('(3)')

# Size of the array
n = 50

# List of number of randomly generated samples
# (Basically number of permutations of array A[1 to n])
sizeList = [100, 1000, 10000]

# Iterating over all the sample sizes
for i in sizeList:

    # List that provides random permutation of array A[1 to n]
    a = [np.random.permutation(np.arange(n)) for x in range(i)]
    
    # Counter to store number of deranged arrays out of i
    ans = 0
    
    # Iterating over all i randomly generated permuted arrays
    for p in range(i):
        
        # List that shows entries where a[x] = x
        sameIndAndVal = list(filter(lambda x: a[p][x] == x, range(n)))

        # If no such element exists, the array is deranged
        if(len(sameIndAndVal) == 0):
            ans += 1
            
    # Printing the predicted value of 'e'
    # It's basically the reciprocal of probability of getting 
    # at least one deranged array out of i
    print('  e \u2248 '+ str(round(i/ans, 5)) +' (Sample Size = '+ str(i) + ')')