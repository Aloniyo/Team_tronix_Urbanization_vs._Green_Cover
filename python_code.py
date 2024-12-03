import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load datasets 
data = pd.read_csv('data.csv')

# Data preprocessing (handle missing values, outliers, etc.)
# Exclude non-numeric columns before applying mean() for fillna
numeric_cols = data.select_dtypes(include=[np.number]).columns
# Fill missing values with the mean of each numeric column
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Check for any remaining missing values
if data.isnull().sum().any():
    print("Warning: There are still missing values after filling with mean.")
else:
    print("No missing values in the dataset after preprocessing.")

# Feature selection (urban growth vs green cover)
X = data[['population_density', 'built_up_area']]  # Example features for urban growth
y = data['green_cover_percentage']  # Target variable: percentage of green cover

# Check data types to ensure correct numeric values
print("\nData types and missing values check:")
print(data.dtypes)
print(data.isnull().sum())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model with different metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Mean Absolute Error: {mae}")

# Cross-validation for more reliable performance metrics
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"\nCross-validated Mean Squared Error: {-cv_scores.mean()}")

# Plot the predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Green Cover Percentage')
plt.ylabel('Predicted Green Cover Percentage')
plt.title('Actual vs Predicted Green Cover Percentage')
plt.show()
plt.savefig('OUTPUT_01.png')
# Visualizing the data - Urban growth vs Green cover
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='population_density', y='green_cover_percentage', palette='viridis')
plt.title('Population Density vs Green Cover Percentage')
plt.xlabel('Population Density')
plt.ylabel('Green Cover Percentage')
plt.legend()
plt.show()
plt.savefig('OUTPUT_02.png')
# Correlation heatmap to explore relationships between variables
corr = data[['population_density', 'built_up_area', 'green_cover_percentage']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
plt.savefig('OUTPUT_03.png')
# Feature Importance from Random Forest
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Visualizing Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance from Random Forest')
plt.show()
plt.savefig('OUTPUT_04.png')
# Pairplot to visualize relationships between variables
sns.pairplot(data[['population_density', 'built_up_area', 'green_cover_percentage']], kind='reg')
plt.show()
plt.savefig('OUTPUT_05.png')
# Display dataset info (check data types and number of entries)
print("\nDataset Info:")
print(data.info())

# Step 4: Visualizations for trends over time
plt.figure(figsize=(15, 10))

# Total Population Trend
plt.subplot(2, 3, 1)
plt.plot(data['Year'], data['Total Population'], marker='o', color='blue', label='Total Population')
plt.title('Total Population Trend', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Total Population', fontsize=10)
plt.grid(True)
plt.legend()

# Urban Population Growth
plt.subplot(2, 3, 2)
plt.plot(data['Year'], data['Urban Population'], marker='o', color='green', label='Urban Population')
plt.title('Urban Population Growth', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Urban Population', fontsize=10)
plt.grid(True)
plt.legend()

# Urban Area Expansion
plt.subplot(2, 3, 3)
plt.plot(data['Year'], data['Urban Area (sq km)'], marker='o', color='orange', label='Urban Area (sq km)')
plt.title('Urban Area Expansion', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Urban Area (sq km)', fontsize=10)
plt.grid(True)
plt.legend()

# Green Cover Area Changes
plt.subplot(2, 3, 4)
plt.plot(data['Year'], data['Green Cover Area (sq km)'], marker='o', color='purple', label='Green Cover Area (sq km)')
plt.title('Green Cover Area Changes', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Green Cover Area (sq km)', fontsize=10)
plt.grid(True)
plt.legend()

# Green Cover Percentage Trend
plt.subplot(2, 3, 5)
plt.plot(data['Year'], data['green_cover_percentage'], marker='o', color='red', label='Green Cover Percentage')
plt.title('Green Cover Percentage Trend', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Green Cover Percentage (%)', fontsize=10)
plt.grid(True)
plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
plt.savefig('OUTPUT_06.png')