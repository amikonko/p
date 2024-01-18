# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset = pd.read_csv("data_with_SSx.csv")
# Assuming 'dataset' is your DataFrame
X = dataset.drop(["id_original", "Number_of_vendors"], axis=1).values
Y = dataset["Number_of_vendors"]
print(Y)
print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
import numpy as np
# Convert y_pred and y_test to NumPy arrays
y_pred_np = np.array(y_pred)
y_test_np = np.array(y_test)
print(np.concatenate((y_pred_np.reshape(len(y_pred_np), 1), y_test_np.reshape(len(y_test_np), 1)), axis=1))

#lets check how good it is
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)










