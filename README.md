# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Ramanujam 
RegisterNumber:  212224240129
*/
```

## Output: 
## Dataset

<img width="221" height="764" alt="image" src="https://github.com/user-attachments/assets/ec4e7ef6-bbee-46eb-8cba-e8389cdcad5c" />

## Head

<img width="209" height="172" alt="image" src="https://github.com/user-attachments/assets/56f3534c-d22c-424c-befe-41c58fe0abff" />

## Tail values

<img width="220" height="178" alt="image" src="https://github.com/user-attachments/assets/d9709bd7-ca1e-43ee-85b8-82ec20fddaba" />

## X and Y values

<img width="953" height="797" alt="image" src="https://github.com/user-attachments/assets/de16e1e7-fee4-41c4-a7ce-ac4cc98e7565" />

## Predication values of X and Y

<img width="946" height="93" alt="image" src="https://github.com/user-attachments/assets/6fd0ab70-59de-4339-9207-e48bc86037ad" />

## MSE,MAE and RMSE

<img width="326" height="113" alt="image" src="https://github.com/user-attachments/assets/b0b83428-c041-41aa-8c89-cebfa535de4c" />

## Training Set

<img width="596" height="443" alt="image" src="https://github.com/user-attachments/assets/21e927ca-aeb1-42ea-bc1a-3649dde481d6" />

## Training Set

<img width="616" height="432" alt="image" src="https://github.com/user-attachments/assets/e95c5a5f-73f7-45bb-96e3-2edb464fbaf2" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
