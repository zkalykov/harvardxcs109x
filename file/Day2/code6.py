import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# Read the data from the file "Advertising.csv"
data_filename = 'advertising.csv'
df = pd.read_csv(data_filename)


# Set 'TV' as the predictor variable (independent variable)
x = df[['TV']]
# print("value of x:" ,x)

# Set 'Sales' as the response variable (dependent variable)
y = df['Sales']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=42)

# Define k_list with possible k values from 1 to 70
k_list = np.arange(1, 71)


# Set the grid to plot the values
fig, ax = plt.subplots(figsize=(10,6))

# Loop over all the k values to find and plot predictions for k=1, 10, 70
for k_value in k_list:
    # Create and fit the kNN regression model
    model = KNeighborsRegressor(n_neighbors=k_value)
    model.fit(x_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(x_test)
    
    # Plot only for k=1, 10, 70
    if k_value in [1, 10, 70]:
        # Create a range of values for x to predict on
        x_vals = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_preds = model.predict(x_vals)
        
        # Plot the model predictions
        ax.plot(x_vals, y_preds, label=f'k = {k_value}')

# Display the original training data points
ax.scatter(x_train, y_train, color='black', label='Training data')

# Set the title and axis labels
ax.set_title('k-Nearest Neighbors Regression')
ax.set_xlabel('TV Budget in $1000')
ax.set_ylabel('Sales in $1000')
ax.legend()

# Show the plot
plt.show()
