
# 3) Scenario: Predicting House Prices

# Imagine you are a data scientist tasked with predicting house prices based on various features of the house. The dataset you're working with includes the following features:

#     Size (sqft): The size of the house in square feet.
#     Bedrooms: The number of bedrooms in the house.
#     Bathrooms: The number of bathrooms in the house.
#     Age: The age of the house in years.
#     Location: The location of the house, coded as 0 for "suburban", 1 for "urban", and 2 for "rural".

# Your goal is to build a Linear Regression model that can predict the Price of a house based on these features.
# Sample Data

# Here's a small sample from your dataset:
# Size (sqft)	Bedrooms	Bathrooms	Age	Location	Price ($)
# 2200	4	3	5	1	500000
# 1650	3	2	12	2	350000
# 3000	5	4	4	1	650000
# 2100	3	3	10	0	400000
# 1200	2	1	15	2	300000


# Task Steps

#     Preprocess the Data: Ensure the data is suitable for Linear Regression, which might include converting categorical variables (like "Location") into dummy variables if you're using a method that requires purely numerical input.

#     Split the Data: Divide the dataset into a training set and a testing set.

#     Build the Linear Regression Model: Use the training set to train your Linear Regression model.

#     Evaluate the Model: Assess the model's performance using the testing set, typically by calculating the Root Mean Squared Error (RMSE) or R-squared value.

# Solution:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

data = {
    'Size_sqft': [2200, 1650, 3000, 2100, 1200],
    'Bedrooms': [4, 3, 5, 3, 2],
    'Bathrooms': [3, 2, 4, 3, 1],
    'Age': [5, 12, 4, 10, 15],
    'Location': [1, 2, 1, 0, 2],
    'Price': [500000, 350000, 650000, 400000, 300000]
}
df = pd.DataFrame(data)

X = df[['Size_sqft', 'Bedrooms', 'Bathrooms', 'Age', 'Location']] 
y = df['Price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
