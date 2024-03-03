# HarvardX CS109x: Data Science - Course Journal
<img src="https://edx-cdn.org/v3/prod/logo.svg" width="100" height="auto">


## Daily Logs

### Day 1 - 02/26/2024
- Finished introduciton section, started first.
- downloaded numpy,python,pandas
- Learned introduction into KNN, Prediction
- #csv #python #metplatlib #numpy #pandas
- All codes maybe found at [https://github.com/zkalykov/harvardxcs109x/tree/main/file/Day1](https://github.com/zkalykov/harvardxcs109x/tree/main/file/Day1)

### Day 2 - 02/27/2024
- Introduction to Regression
- Worked with KNN
- Trained
- codes at [https://github.com/zkalykov/harvardxcs109x/tree/main/file/Day2](https://github.com/zkalykov/harvardxcs109x/tree/main/file/Day2)
  
### Day 3 - 02/28/2024
- Still exploring KNN regression
Summary: DataFrame and series and difference is 2d dimension and 1d dimension.
KNN algorithm in total gives us the value of nearest approriate value according K neighbours.
- knn in general finds the best predicted value.
- : It could be fonund as following:
- Lets say we have a tv budget with 150 and k=3 // which is k is better for smaller datta, we will go for all sales, and find distance, let's say we have 1....n data
- and we will find difference which is dif[0-n] and the knn result will be sum(dif[0] to dif[n-1])/k // we will delete to k
  
- codes at [https://github.com/zkalykov/harvardxcs109x/tree/main/file/Day3](https://github.com/zkalykov/harvardxcs109x/tree/main/file/Day3)

### Day 4 - 02/29/2024
  - Exercise: Simple kNN Regression
  - Task 1 optimization
  - Overall, I need practice more, regardig task 1, I have optimized code, and it gives more correct results, code can be found trough link

### Day 5 -02/01/2024
- Using sklearn package, 
- x = np.linspace(np.min(x_true), np.max(x_true)) - finds minimum and maximum of x values and will create array of evenly spaced values between them
- y = np.zeros((len(x))) - fills | 0s
- Training
- Practice
### Day 6 -02/02/2024
- Training KNN | Solved task Engine/Sales prediction with simple knn by manuel k=3 .
  ```
  import pandas as pd 
  import matplotlib.pyplot as plt 
  from sklearn.model_selection import train_test_split
  from sklearn.neighbors import KNeighborsRegressor
  df=pd.read_csv('task.csv')
  x=df[['EngineSize']]
  y=df['Price']
  x_train,x_test, y_train,y_test = train_test_split ( x,y,test_size = 0.2, random_state=1 )
  knn = KNeighborsRegressor(n_neighbors=1)
  knn.fit(x_train,y_train)
  #  1 to 30
  x_list=[]
  y_list=[]
  i=1
  while i<=4:
      j = knn.predict([[i]])
      x_list.append(i)
      y_list.append(j)
      i+=0.5
  plt.plot(x_list,y_list,'-.',color='green',label="predicted for 10")
  
  plt.scatter(x,y,color='red',label='Actual data')
  plt.show()
```


