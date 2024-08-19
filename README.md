# Project-1
This repository contains the code and resources for a project focused on analyzing and optimizing crop production. The project utilizes data analysis and machine learning techniques to predict crop yields and provide recommendations for improving agricultural practices.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Load the dataset
df = pd.read_csv('crop_production_data.csv')

# Display the first few rows and summary of the dataset
print("First few rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values (or handle them as necessary)
df = df.dropna()

# Convert data types if needed
df['Crop_Year'] = df['Crop_Year'].astype(int)

# Visualization: Total production by crop
plt.figure(figsize=(12, 6))
sns.barplot(x='Crop', y='Production', data=df.groupby('Crop')['Production'].sum().reset_index().sort_values('Production', ascending=False))
plt.xticks(rotation=90)
plt.title('Total Production by Crop')
plt.xlabel('Crop')
plt.ylabel('Total Production')
plt.show()

# Visualization: Total production by year
plt.figure(figsize=(12, 6))
sns.lineplot(x='Crop_Year', y='Production', data=df.groupby('Crop_Year')['Production'].sum().reset_index())
plt.title('Total Production by Year')
plt.xlabel('Year')
plt.ylabel('Total Production')
plt.show()

# Visualization: Production by state
plt.figure(figsize=(12, 6))
sns.barplot(x='State_Name', y='Production', data=df.groupby('State_Name')['Production'].sum().reset_index().sort_values('Production', ascending=False))
plt.xticks(rotation=90)
plt.title('Total Production by State')
plt.xlabel('State')
plt.ylabel('Total Production')
plt.show()

# Visualization: Production by district for a specific state
state = 'Andaman and Nicobar Islands'
df_state = df[df['State_Name'] == state]
plt.figure(figsize=(12, 6))
sns.barplot(x='District_Name', y='Production', data=df_state.groupby('District_Name')['Production'].sum().reset_index().sort_values('Production', ascending=False))
plt.xticks(rotation=90)
plt.title(f'Total Production by District in {state}')
plt.xlabel('District')
plt.ylabel('Total Production')
plt.show()

# Feature Engineering: Adding total area under cultivation
df['Total_Area'] = df.groupby('Crop_Year')['Area'].transform('sum')

# Prepare features and target for predictive modeling
X = df[['Area', 'Crop_Year']]  # Include other relevant features if available
y = df['Production']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print('\nMean Squared Error:', mean_squared_error(y_test, y_pred))

# Create interactive dashboard visualization with Plotly
fig = px.line(df, x='Crop_Year', y='Production', title='Total Production by Year')
fig.show()


