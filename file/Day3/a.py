import matplotlib.pyplot as plt

# Original data points
tv_budgets = [230.1, 44.5, 17.2, 151.5, 180.8]
sales = [22.1, 10.4, 9.3, 18.5, 12.9]

# Hypothetical predicted data point
predicted_tv = 150
predicted_sales = 15

# Plotting the original data points
plt.scatter(tv_budgets, sales, color='blue', label='Original Data')

# Plotting the predicted data point
plt.scatter(predicted_tv, predicted_sales, color='red', label='Predicted Data', marker='x')

# Adding plot labels and title
plt.xlabel('TV Budget in $1000')
plt.ylabel('Sales in $1000')
plt.title('TV Budget vs. Sales with Predicted Value')
plt.legend()

# Showing the plot
plt.show()
