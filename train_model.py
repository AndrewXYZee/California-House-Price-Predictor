#Import required libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib

#Load and convert data to DataFrame
cali = fetch_california_housing()
df = pd.DataFrame(cali.data, columns=cali.feature_names)
df['PRICE'] = cali.target

#Clean data - remove extreme values
df = df[df['PRICE'] <= 4.8]
df = df[df['AveRooms'] <= 40]
df = df[df['AveBedrms'] <= 12]
df = df[df['Population'] <= 17000]
df = df[df['AveOccup'] <= 20]

#Log-transform right-skewed features
df['log_MedInc'] = np.log1p(df['MedInc'])
df['log_AveRooms'] = np.log1p(df['AveRooms'])
df['log_AveBedrms'] = np.log1p(df['AveBedrms'])
df['log_AveOccup'] = np.log1p(df['AveOccup'])

#Create relation features
df['rooms_per_person']= df['AveRooms'] / df['AveOccup'] # rooms per person

#Load prepared data
X = df.drop('PRICE', axis=1)
y = df['PRICE']

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f} ($100k)")
print(f"R2: {r2:.4f}")

#Save the model
joblib.dump(model, 'cali.joblib')