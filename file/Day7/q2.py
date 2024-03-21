
# 2) Scenario: Predicting Customer Churn for a Telecom Company

# A telecom company is interested in predicting customer churn based on several features of customer activity and demographic information. Your task is to use KNN to predict whether a customer will churn (1 for churn, 0 for no churn) based on the following features:

#     Tenure: Number of months the customer has been with the company.
#     MonthlyCharges: The amount charged to the customer each month.
#     TotalCharges: The total amount charged to the customer.
#     Contract: Type of contract (coded as: 0 for Month-to-month, 1 for One year, 2 for Two year).
#     OnlineSecurity: Whether the customer has online security (coded as: 0 for No, 1 for Yes).

# Sample Data

# Here's a small sample from the dataset:
# Tenure	MonthlyCharges	TotalCharges	Contract	OnlineSecurity	Churn
# 1	29.85	29.85	0	0	0
# 34	56.95	1889.5	1	1	0
# 2	53.85	108.15	0	1	1
# 45	42.3	1840.75	1	0	0
# 2	70.7	151.65	0	0	1
# 8	99.65	820.5	0	0	1
# 22	89.1	1949.4	1	1	0
# 10	29.75	301.9	0	0	0

# Task
#     Preprocess the Data:
#         Handle any missing values if necessary.
#         Convert TotalCharges from a string to a numeric type if working with the actual dataset.
#         Scale the features using standardization or normalization.

#     Split the Data: Divide the dataset into training and testing sets.
#     Implement KNN: Use KNN to predict customer churn. Experiment with different values of k to find the most effective configuration.
#     Evaluate the Model: Assess the model's performance using the accuracy score or any other relevant metric.

# Solution:

data = {
    'Tenure': [1, 34, 2, 45, 2, 8, 22, 10],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.3, 70.7, 99.65, 89.1, 29.75],
    'TotalCharges': [29.85, 1889.5, 108.15, 1840.75, 151.65, 820.5, 1949.4, 301.9],
    'Contract': [0, 1, 0, 1, 0, 0, 1, 0],
    'OnlineSecurity': [0, 1, 1, 0, 0, 0, 1, 0],
    'Churn': [0, 0, 1, 0, 1, 1, 0, 0]
}
new_customer = {'Tenure': 12, 'MonthlyCharges': 50, 'TotalCharges': 600, 'Contract': 0, 'OnlineSecurity': 0}
def euclidean_distance(customer1, customer2):
    distance = 0
    for key in customer1:
        if key in customer2:
            distance += (customer1[key] - customer2[key]) ** 2
    return distance ** 0.5

customers = [dict(zip(data, t)) for t in zip(*data.values())]

distances = []
for customer in customers:
    dist = euclidean_distance(new_customer, customer)
    distances.append((dist, customer['Churn']))

k = 3
distances.sort(key=lambda x: x[0])
nearest_neighbors = distances[:k]

votes = [neighbor[1] for neighbor in nearest_neighbors]
prediction = round(sum(votes) / k)

print(f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
