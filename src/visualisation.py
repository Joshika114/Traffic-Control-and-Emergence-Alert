import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Assume y_test, y_pred_dt, y_pred_knn, X_train, dt_model are defined

# --------------------------------------------
# 1. Residual Plots
# --------------------------------------------
residuals_dt = y_test - y_pred_dt
residuals_knn = y_test - y_pred_knn

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_dt, y=residuals_dt, color='royalblue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Decision Tree Residual Plot')
plt.xlabel('Predicted Total Time')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_knn, y=residuals_knn, color='seagreen')
plt.axhline(0, color='red', linestyle='--')
plt.title('KNN Residual Plot')
plt.xlabel('Predicted Total Time')
plt.ylabel('Residuals')
plt.tight_layout()
plt.show()

# 🔍 Interpretation: Residuals near zero & no clear pattern = good model fit

# --------------------------------------------
# 2. Predicted vs Actual
# --------------------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_dt, color='dodgerblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Decision Tree: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_knn, color='mediumseagreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('KNN: Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.tight_layout()
plt.show()

# 🔍 Interpretation: Closer to diagonal = more accurate predictions

# 🔍 Interpretation: Higher bars = more influence on total_time

# --------------------------------------------
# 4. Performance Comparison (MAE, RMSE, R²)
# --------------------------------------------
models = ['Decision Tree', 'KNN']
mae_scores = [mean_absolute_error(y_test, y_pred_dt), mean_absolute_error(y_test, y_pred_knn)]
rmse_scores = [np.sqrt(mean_squared_error(y_test, y_pred_dt)), np.sqrt(mean_squared_error(y_test, y_pred_knn))]
r2_scores = [r2_score(y_test, y_pred_dt), r2_score(y_test, y_pred_knn)]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, mae_scores, width, label='MAE', color='cornflowerblue')
plt.bar(x, rmse_scores, width, label='RMSE', color='lightcoral')
plt.bar(x + width, r2_scores, width, label='R² Score', color='mediumseagreen')

plt.xticks(x, models)
plt.ylabel('Error / Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# 🔍 Interpretation:
# - Lower MAE and RMSE = better error performance
# - Higher R² = better explained variance
