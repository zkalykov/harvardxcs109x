import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('credit.csv')
print(df.head())

x = df.drop('Balance', axis=1)
y = df['Balance']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

try:
    test_model = LinearRegression().fit(x_train, y_train)
except Exception as e:
    print('Error:', e)

numeric_features = ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']
model1 = LinearRegression().fit(x_train[numeric_features], y_train)

train_score_model1 = model1.score(x_train[numeric_features], y_train)
test_score_model1 = model1.score(x_test[numeric_features], y_test)
print(f'Model 1 Train R^2: {train_score_model1}, Test R^2: {test_score_model1}')

x_train_design = pd.get_dummies(x_train, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)
x_test_design = pd.get_dummies(x_test, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)

model2 = LinearRegression().fit(x_train_design, y_train)

train_score_model2 = model2.score(x_train_design, y_train)
test_score_model2 = model2.score(x_test_design, y_test)
print(f'Model 2 Train R^2: {train_score_model2}, Test R^2: {test_score_model2}')

best_cat_feature = 'Student_Yes'
features = ['Income', best_cat_feature]

model3 = LinearRegression().fit(x_train_design[features], y_train)

beta0, beta1, beta2 = model3.intercept_, *model3.coef_

coefs = pd.DataFrame([beta0, beta1, beta2], index=['Intercept'] + features, columns=['beta_value'])
sns.barplot(data=coefs.T, orient='h').set(title='Model 3 Coefficients')

x_space = np.linspace(x['Income'].min(), x['Income'].max(), 1000)
y_hat_yes = beta0 + beta1 * x_space + beta2 * 1  # Scenario where the categorical feature is true
y_hat_no = beta0 + beta1 * x_space + beta2 * 0   # Scenario where the categorical feature is false

ax = sns.scatterplot(x=x_train_design['Income'], y=y_train, hue=x_train_design[best_cat_feature], alpha=0.8, palette='viridis')
ax.plot(x_space, y_hat_yes, label='With Best Feature')
ax.plot(x_space, y_hat_no, label='Without Best Feature')
plt.legend()
plt.title('Predicted Balance Across Income Levels')
plt.show()
