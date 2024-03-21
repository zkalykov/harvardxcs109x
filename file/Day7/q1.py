# 1) Scenario: Predicting Loan Approval

# Imagine you're working with a dataset from a bank, and your task is to predict whether a loan application will be approved based on two features: Applicant's Income (in thousands) and Credit Score. The outcome is binary: 1 for approved and 0 for not approved.
# Sample Data:
# Applicant Income (in thousands)	Credit Score	Loan Approved
# 50	700	1
# 20	600	0
# 30	650	0
# 80	800	1
# 40	670	1
# 70	720	1
# 25	580	0
# Task: Your task is to predict whether a loan will be approved for an applicant with an income of 45 thousand and a credit score of 690 using KNN with k=3.

# Solution:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np 
income=np.array([50, 20, 30, 80, 40, 70, 25])
credit=np.array([700, 600, 650, 800, 670, 720, 580])
approved= np.array([1, 0, 0, 1, 1, 1, 0])

def e_dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

new_a = [45,690]

distance=[]
for i in range(len(income)):
    d = e_dist(income[i],credit[i],new_a[0],new_a[1])
    distance.append(d)
distance=np.array(distance)

nearest_neighbors = distance.argsort()[:3]

prediction = approved[nearest_neighbors].mean()
print(prediction)
