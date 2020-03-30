# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing our data set
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values # x should always be a matrix try to change it to [:,1] you will see the differnece
y=dataset.iloc[:,2].values



# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scalling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y=StandardScaler()
# you can scale your dummy variables or not based on the context and what you want to do
# with your data if you scalled it you will lose the knoweldge of the encding 
# but you may get better accuracy if you don't they will be already scaled for this model
# we will scale them
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train=np.squeeze(sc_y.fit_transform(y_train.reshape(-1, 1)))
y_test=np.squeeze(sc_y.transform(y_test.reshape(-1, 1)))
"""

# Fitting Random Forest Regression to the dataset
# create your regressor
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

# Predicting a new result with the Random Forest Regression
y_pred=regressor.predict([[6.5]])

# Visualizing the Random Forest Regression with higher resolution and smoother curve
X_grid=np.arange(min(X),max(X),0.01) # increment by 0.1 to get better curve
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
