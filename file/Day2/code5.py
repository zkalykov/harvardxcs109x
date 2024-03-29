import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'advertising.csv'
df_adv = pd.read_csv(filename)

x_subset = df_adv['TV'].iloc[5:14].values
y_subset = df_adv['Sales'].iloc[5:14].values

sorted_indices = np.argsort(x_subset)
x_sorted = x_subset[sorted_indices]
y_sorted = y_subset[sorted_indices]

def find_nearest_neighbor(x_query, x_data, y_data):
    distances = np.abs(x_data - x_query)
    nearest_index = np.argmin(distances)
    return y_data[nearest_index] 

x_values = np.linspace(x_sorted.min(), x_sorted.max(), 300)

y_values = np.array([find_nearest_neighbor(x, x_sorted, y_sorted) for x in x_values])

# Plotting the kNN predictions
plt.plot(x_values, y_values, linestyle='-.', label='kNN Predictions', color='blue')

# Plotting the original subset data
plt.scatter(x_sorted, y_sorted, color='black', label='Original Data')

# Set the title and axis labels
plt.title('TV vs Sales (k=1)')
plt.xlabel('TV Budget in $1000')
plt.ylabel('Sales in $1000')
plt.legend()

# Show the plot
plt.show()
