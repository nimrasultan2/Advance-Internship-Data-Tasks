import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the correct dataset
df = pd.read_csv("credit_risk_dataset.csv")

# 2. Preprocessing tailored to YOUR data
# Handle missing values (common in employment length and interest rate)
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# One-hot encode categorical features (Home ownership, Intent, etc.)
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

# 3. Define Features and Target
# In your file, 'loan_status' is the target (1 = default, 0 = paid)
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Train the Model
model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Business Cost Optimization
# FP: Rejecting a good loan (Lost interest)
# FN: Accepting a bad loan (Lost principal)
COST_FP = 500  
COST_FN = 5000 

y_probs = model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 100)
costs = []

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = (fp * COST_FP) + (fn * COST_FN)
    costs.append(total_cost)

best_t = thresholds[np.argmin(costs)]

# 6. Results
print(f"Optimal Threshold for Business: {best_t:.4f}")
print(f"Minimum Total Cost: ${min(costs):,}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs, color='blue', label='Total Business Cost')
plt.axvline(best_t, color='red', linestyle='--', label=f'Best Threshold ({best_t:.2f})')
plt.title('Business Cost Optimization')
plt.xlabel('Probability Threshold')
plt.ylabel('Cost ($)')
plt.legend()
plt.show()