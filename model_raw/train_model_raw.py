#Import required libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

#Load build-in California housing dataset
cali = fetch_california_housing()
X = cali.data
y = cali.target

#Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f} ($100k)")
print(f"R2: {r2:.4f}")

#Save the model
joblib.dump(model, 'cali_raw.joblib')
