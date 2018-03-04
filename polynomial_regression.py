# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # X всегда должна быть матрица (1:2) fake
y = dataset.iloc[:, 2].values # y всегда должен быть вектор

# так как только 10 строчек информации, мы берем максимум информации
# иначе будет не особо точная модель, а нужна точная максимально
# плюс данные последовательные, поэтому берем всю модель!
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
# создаем для сравнения работы моделей
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 2) # ^2
X_poly = poly_regressor.fit_transform(X) # почему fit и transform??? l55 7:30
# X_poly мы создали имкуственно polynomial ^2 данные из данных X
# 1 3 9   1 4 16   1 5 25 ...
lin_regressor_2 = LinearRegression() # используем обычную новую линейную модель
lin_regressor_2.fit(X_poly, y) # закидываем в линейную регрессию новые данные


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regressor_2.predict(poly_regressor.fit_transform(X)), color = 'blue')
# Важно!!! в .predict можно закинуть и X_poly, но при этом у нас данные будут
# только для X_poly, так как нам нужен определенный формат данных,
# и это не удобно, лучше взять 37 строку,
# что бы можно было на ходу менять занчения X в трех местах
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

















