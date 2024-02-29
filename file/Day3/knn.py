import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Prepare your dataset
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(df[['TV']], df['Sales'])
predicted_sales_sklearn = knn.predict([[150]])
print("predicted_sales_sklearn: ",predicted_sales_sklearn)




plt.scatter(df['TV'], df['Sales'], color='blue', label='Original Data')
plt.scatter(150, predicted_sales_sklearn, color='green', label='Predicted Data (sklearn)', marker='x')

plt.xlabel('TV Budget in $1000')
plt.ylabel('Sales in $1000')
plt.title('TV Budget vs. Sales with sklearn KNN Prediction')
plt.legend()
plt.show()
