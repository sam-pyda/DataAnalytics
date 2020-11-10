# DataAnalytics
Spark Foundation (NOV20)
PROGRAMMER: STEPHEN, SAMUEL OCHOGBE

Objective :Predicting the percentage scores of students on the number of study hours using supervised ML.
Tool used: Python(Jupyter Notebook)

Code:
### IMPORTING ALL LIBRARIES REQUIRED

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

### READING DATA FROM REMOTE LINK

url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
s_data = pd.read_csv(url)
print('imported successfully')

s_data.head(11)

### 2-D PLOT OF THE DISTRIBUTION OF SCORES

s_data.plot(x='Hours', y='Scores', style ='X')
plt.title('Hours vs perventage')
plt.xlabel('Hours studied')
plt.ylabel('percentage score')
plt.show()

from the graph we see that there's a positive relationship between the hours and percentage scores.

### PREPARING DATA

X= s_data.iloc[:, :-1].values
y= s_data.iloc[:, 1].values
print(y)


Spliting data into train and test sets from the scikit-learn's train_test_split() method

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state =0)

### TRAINING THE ALGORITHM

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('training complete')

### PLOTTING REGRESSION LINE

line = regressor.coef_*X + regressor.intercept_
plt.scatter(X,y)
plt.plot(X, line);
plt.show()

### MAKING PREDICTIONS

print(X_test)
y_pred = regressor.predict(X_test)
print(y_pred)
print(y_test)


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 

### MY prediction for 9.25 hrs

#You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

### EVALUATING THE MODEL

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

The mean error shows the marginal difference between the tested and predicted dataset, which reflecs the performance of different algorithm on a dataset.
