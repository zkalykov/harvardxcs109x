import pandas as pd
import matplotlib.pyplot as plt

# Reading data from CSV
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)

# Function to calculate Euclidean distance between two points
def euclidean_distance(tv1, tv2):
    return abs(tv1 - tv2)

# Function to find the k nearest neighbors to a given query point
def find_k_nearest_neighbors(df, query_tv, k):
    distances = []
    for i in range(len(df)):
        dist = euclidean_distance(query_tv, df.iloc[i]['TV']) #150 - tv 
        distances.append((df.iloc[i]['Sales'], dist))
    
    for i in distances:
        print(i)
   
    
    distances.sort(key=lambda x: x[1])
    print("AFTER SORT:")
    for i in distances:
        print(i)
    
    neighbors = distances[:k]
    return neighbors

# Function to predict the target value for a given query point using KNN
def knn_predict(df, query_tv, k):
    neighbors = find_k_nearest_neighbors(df, query_tv, k)
    total_sales = 0
    for neighbor in neighbors:
        total_sales += neighbor[0]  # Add each neighbor's sales to the total
    average_sales = total_sales / k  # Calculate the average sales based on k neighbors
    return average_sales

# Example usage
query_tv = 150  # Example query point (TV budget)
k = 3  # Number of neighbors to consider
predicted_sales = knn_predict(df, query_tv, k)

# Plotting the original data points
plt.scatter(df['TV'], df['Sales'], color='blue', label='Original Data')

# Plotting the predicted data point
plt.scatter(query_tv, predicted_sales, color='red', label='Predicted Data', marker='x')

# Adding plot labels and title
plt.xlabel('TV Budget in $1000')
plt.ylabel('Sales in $1000')
plt.title('TV Budget vs. Sales with KNN Prediction')
plt.legend()

# Showing the plot
plt.show()
