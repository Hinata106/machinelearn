import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Load the dataset
dataset = pd.read_csv("HousePricePrediction.csv")

# Display basic info about the dataset
print(dataset.head())
print(dataset.shape)

# Identify categorical and numeric columns
object_cols = dataset.select_dtypes(include=['object']).columns
num_cols = dataset.select_dtypes(include=['int', 'float']).columns

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(dataset[num_cols].corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Visualize unique values of categorical columns
unique_values = dataset[object_cols].nunique()
plt.figure(figsize=(10, 6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=unique_values.index, y=unique_values.values)
plt.show()

# Visualize distribution of categorical features
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
for i, col in enumerate(object_cols, 1):
    plt.subplot(11, 4, i)
    plt.xticks(rotation=90)
    sns.countplot(data=dataset, x=col)
plt.tight_layout()
plt.show()

# Drop 'Id' column and handle missing values
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'].fillna(dataset['SalePrice'].mean(), inplace=True)
new_dataset = dataset.dropna()

# One-hot encode categorical variables
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names(object_cols)
df_final = pd.concat([new_dataset[num_cols], OH_cols], axis=1)

# Prepare data for modeling
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# SVR model
model_SVR = SVR()
model_SVR.fit(X_train, Y_train)
Y_pred = model_SVR.predict(X_valid)
mape_SVR = mean_absolute_percentage_error(Y_valid, Y_pred)
print("MAPE for SVR:", mape_SVR)

# Random Forest model
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
mape_RFR = mean_absolute_percentage_error(Y_valid, Y_pred)
print("MAPE for Random Forest:", mape_RFR)

# Linear Regression model
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
mape_LR = mean_absolute_percentage_error(Y_valid, Y_pred)
print("MAPE for Linear Regression:", mape_LR)
