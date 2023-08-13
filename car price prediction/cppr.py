import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("carprice.csv")
data.head()
data.shape
data.isnull().sum()
#So this dataset doesn’t have any null values, now let’s look at some of the other important insights to get 
#an idea of what kind of data we’re dealing with:
data.info()
data.describe()

# Drop the 'carname' column
data = data.drop(columns=['carname'])

sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.histplot(data.price, kde=True)
plt.show()
'''
# Now let’s have a look at the correlation among all the features of this dataset:
print(data.corr())

plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()'''

#Training a Car Price Prediction Model
predict = "price"

# Data preprocessing
selected_features = ["symboling", "wheelbase", "carlength", 
                     "carwidth", "carheight", "curbweight", 
                     "enginesize", "boreratio", "stroke", 
                     "compressionratio", "horsepower", "peakrpm", 
                     "citympg", "highwaympg"]

X = data[selected_features]
y = data["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Example prediction
example_data = np.array([[0, 95.1, 158.7, 63.6, 52.0, 2017, 141, 3.78, 3.15, 8.0, 95, 5000, 27, 34]])
example_df = pd.DataFrame(example_data, columns=selected_features)
predicted_price = model.predict(example_df)
print("Predicted Car Price:", predicted_price[0])

print(predictions)
