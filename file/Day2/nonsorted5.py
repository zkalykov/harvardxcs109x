import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the file 'advertising.csv' is correctly loaded
filename = 'advertising.csv'
df_adv = pd.read_csv(filename)

# Selecting a subset without sorting
x_subset = df_adv['TV'] 
y_subset = df_adv['Sales'] 

def find_nearest_neighbor(x_query, x_data, y_data):
    distances = np.abs(x_data - x_query)
    nearest_index = np.argmin(distances)
    return y_data[nearest_index]

# Generating a range of x values for plotting the kNN predictions
x_values = np.linspace(x_subset.min(), x_subset.max(), 300)

# Finding nearest neighbor predictions for each x value in the range
y_values = np.array([find_nearest_neighbor(x, x_subset, y_subset) for x in x_values])

# Plotting the kNN predictions
plt.plot(x_values, y_values, linestyle='-.', label='kNN Predictions', color='blue')

# Plotting the original subset data without sorting
plt.scatter(x_subset, y_subset, color='black', label='Original Data')

# Setting the title and axis labels
plt.title('TV vs Sales (k=1) Without Sorting')
plt.xlabel('TV Budget in $1000')
plt.ylabel('Sales in $1000')
plt.legend()

# Showing the plot
plt.show()
