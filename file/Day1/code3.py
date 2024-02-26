import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)

predictions = []

for index in range(len(df)):
    train = df.drop(index)
    test = df.iloc[index]
    
    X_train = train.drop('Sales', axis=1)
    y_train = train['Sales']
    
    X_test = test.drop('Sales').to_frame().T
    
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_test)
    
    predictions.append(prediction[0])

df['Predicted Sales'] = predictions
print(df)
