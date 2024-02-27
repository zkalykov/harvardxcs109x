import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

file_to_read = "data.csv"
df = pd.read_csv(file_to_read)

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
