# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# "Advertising.csv" containts the data set used in this exercise
data_filename = 'advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)


# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df['TV'], df['Sales'])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel('TV budget')
plt.ylabel('Sales')

# Add plot title 
plt.title('TV vs Sales Scatter Plot')
# for showing
plt.show()