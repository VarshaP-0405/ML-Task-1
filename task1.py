# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Life Expectancy Data.csv")
print("Sample Data")
print(df.head())

#Explore: Check missing values, basic stats (df.describe()), scatter plots.
print(df.info())
print(df.isnull().sum())
print(df.describe())

# check for missing values and add the mean value in that place
df.fillna(df.mean(numeric_only=True), inplace=True)

#Prepare: Select features (Adult Mortality, Alcohol, GDP, Schooling, HIV/AIDS) and target (Life expectancy), split train/test (80/20).
X = df[['Adult Mortality', 'Alcohol', 'GDP', 'Schooling', 'HIV/AIDS']]
y = df['Life expectancy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model: Use LinearRegression(), fit on training data.
model=LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

 #Evaluate: Predict on test set, calculate Mean Squared Error (MSE), and R² score.
print("Mean Squared Error",mean_squared_error(y_test,y_pred))
print("R2 Score",r2_score(y_test,y_pred))

#Interpret: Check model coefficients to see which factors influence life expectancy most, plot predicted vs actual values.
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coeff_df)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
