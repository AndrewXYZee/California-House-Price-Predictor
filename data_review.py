from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load data
cali = fetch_california_housing()
df = pd.DataFrame(cali.data, columns=cali.feature_names)
df['PRICE'] = cali.target

#Count dataaset rows
original_size = len(df)

#Thresholds
t_price = 4.8
t_averooms = 40
t_avebedrms = 12
t_aveoccup = 20
t_population = 17000

print("Thresholds values:")
print(f"{'Price:':<12}", t_price)
print(f"{'AveRooms:':<12}", t_averooms)
print(f"{'AveBedrms:':<12}", t_avebedrms)
print(f"{'AveOccup:':<12}", t_aveoccup)
print(f"{'Population:':<12}", t_population)

#See thresholds results
print("Removed rows:")

r_price = (df['PRICE'] > t_price).sum() # <-- check which rows are true and convert boolean list to number
p_price = r_price / original_size * 100 # <-- calculate %
print(f"{'Price:':<12}", r_price, "-", round(p_price, 2),"%")

r_averooms = (df['AveRooms'] > t_averooms).sum()
p_averooms = r_averooms / original_size * 100
print(f"{'AveRooms:':<12}", r_averooms, "-", round(p_averooms, 2), "%")

r_avebedrms = (df['AveBedrms'] > t_avebedrms).sum()
p_avebedrms = r_avebedrms / original_size * 100
print(f"{'AveBedrms:':<12}", r_avebedrms, "-", round(p_avebedrms, 2), "%")

r_aveoccup = (df['AveOccup'] > t_aveoccup).sum()
p_aveoccup = r_aveoccup / original_size * 100
print(f"{'AveOccup:':<12}", r_aveoccup, "-", round(p_aveoccup, 2), "%")

r_population = (df['Population'] > t_population).sum()
p_population = r_population  / original_size * 100
print(f"{'Population:':<12}", r_population , "-", round(p_population, 2), "%")

#Boxplot
fig, axes = plt.subplots(3, 3)

sns.boxplot(x=df['MedInc'], ax=axes[0,0])
sns.boxplot(x=df['HouseAge'], ax=axes[0,1])
sns.boxplot(x=df['AveRooms'], ax=axes[0,2])
sns.boxplot(x=df['AveBedrms'], ax=axes[1,0])
sns.boxplot(x=df['Population'], ax=axes[1,1])
sns.boxplot(x=df['AveOccup'], ax=axes[1,2])
sns.boxplot(x=df['Latitude'], ax=axes[2,0])
sns.boxplot(x=df['Longitude'], ax=axes[2,1])
sns.boxplot(x=df['PRICE'], ax=axes[2,2])

plt.tight_layout() #No overlapping text)
plt.show()