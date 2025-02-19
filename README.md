# nba-stats





-------------


## Step 1: Data Collection 

with Player Impact Estimate (PIE) Calculation



## Step 2: Data Preprocessing

Handle any remaining missing values.
Normalize or standardize the data as needed.
Perform feature engineering to create additional useful features.


## Step 3: Model Development

Select appropriate machine learning algorithms.
Train models to identify the best players based on PIE and other statistics.
Evaluate model performance using suitable metrics.



---------------




### 6. Define Success Tiers


Create a categorical target variable based on nPER.

```python
def classify_success(nper):
    if nper >= 18:
        return 'High'
    elif nper >= 15:
        return 'Medium'
    else:
        return 'Low'

merged_df3['Success_Tier'] = merged_df3['nPER'].apply(classify_success)
```


```python
### Encode the target variable

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_class = le.fit_transform(merged_df3['Success_Tier'])
```




### 7. Prepare Data for Classification


```python
X_class = X  # Use the same features
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
```



### 8. Decision Tree Classifier


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and fit the model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_c, y_train_c)

# Predict on test data
y_pred_dt = dt_classifier.predict(X_test_c)

# Evaluate the model
accuracy_dt = accuracy_score(y_test_c, y_pred_dt)
print(f'Decision Tree Accuracy: {accuracy_dt:.2f}')
print(classification_report(y_test_c, y_pred_dt, target_names=le.classes_))
```



### 9. Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and fit the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_c, y_train_c)

# Predict on test data
y_pred_rf = rf_classifier.predict(X_test_c)

# Evaluate the model
accuracy_rf = accuracy_score(y_test_c, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print(classification_report(y_test_c, y_pred_rf, target_names=le.classes_))
```


### 10. Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

# Initialize and fit the model
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_classifier.fit(X_train_c, y_train_c)

# Predict on test data
y_pred_gb = gb_classifier.predict(X_test_c)

# Evaluate the model
accuracy_gb = accuracy_score(y_test_c, y_pred_gb)
print(f'Gradient Boosting Accuracy: {accuracy_gb:.2f}')
print(classification_report(y_test_c, y_pred_gb, target_names=le.classes_))
```




### 11. Neural Network Regressor

```python
from sklearn.neural_network import MLPRegressor

# Initialize and fit the model
nn_regressor = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_regressor.fit(X_train, y_train)

# Predict on test data
y_pred_nn = nn_regressor.predict(X_test)

# Evaluate the model
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
print(f'Neural Network Regression MSE: {mse_nn:.2f}')
print(f'Neural Network Regression R² Score: {r2_nn:.2f}')
```



### 12. Neural Network Classifier


```python
from sklearn.neural_network import MLPClassifier

# Initialize and fit the model
nn_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
nn_classifier.fit(X_train_c, y_train_c)

# Predict on test data
y_pred_nn_c = nn_classifier.predict(X_test_c)

# Evaluate the model
accuracy_nn = accuracy_score(y_test_c, y_pred_nn_c)
print(f'Neural Network Classification Accuracy: {accuracy_nn:.2f}')
print(classification_report(y_test_c, y_pred_nn_c, target_names=le.classes_))
```




## Feature Importance

### 13. Plot Feature Importance from Random Forest

```python
import matplotlib.pyplot as plt
import numpy as np

importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_class.columns

plt.figure(figsize=(12, 6))
plt.title('Feature Importances from Random Forest')
plt.bar(range(len(feature_names)), importances[indices], align='center')
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
```

## Model Evaluation

### 14. Confusion Matrix


```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_c, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
```



### 15. Cross-Validation

Evaluate the stability of your model using cross-validation.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf_classifier, X_class, y_class, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean():.2f}')
```



## Hyperparameter Tuning

### 16. Grid Search for Random Forest

Optimize your model's performance using Grid Search.


```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_c, y_train_c)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_:.2f}')
```





## Next Steps


- Interpret Results: Analyze which features are most important and how they influence NBA success.
- Model Selection: Choose the model that offers the best performance and interpretability.
- Validation: Test your model on a separate validation set or use techniques like bootstrapping.
- Deployment: Consider creating a predictive tool or dashboard to visualize your findings.


