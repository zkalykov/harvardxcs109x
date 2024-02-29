import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

file_to_read = "data.csv"
df = pd.read_csv(file_to_read)

# TV	Radio	Newspaper	Sales
# 230.1	37.8	69.2	22.1
# 44.5	39.3	45.1	10.4
# 17.2	45.9	69.3	9.3
# 151.5	41.3	58.5	18.5
# 180.8	10.8	58.4	12.9
# 38.2	3.7	    13.8	7.6
# 94.2	4.9	    8.1	    9.7
# 177.0	9.3	    6.4	    12.8
# 283.6	42.0	66.2	25.5
# 232.1	8.6	    8.7	    13.4

# Define features and target variable
X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize and train the K-Nearest Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Print the predicted Sales
print("Predicted Sales:", y_pred)

# Create a scatter plot for actual sales vs. predicted sales
plt.scatter(y_test, y_pred, color='red', label='Predicted')



# Create a scatter plot for unused (training) data points
plt.scatter(y_train, y_train, color='black', label='Unused')

# Add scatter plot for used (testing) data points
plt.scatter(y_test, y_test, color='blue', label='Used')



# Add a diagonal line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', label='Actual')

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.show()







