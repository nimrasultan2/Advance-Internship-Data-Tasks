import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(".venv\credit_risk_dataset.csv")

# Fill in missing values using the median so we don't lose rows
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Convert text-based columns into numbers the model can actually work with
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'])

# Separate what we're predicting from everything else
# loan_status: 1 means the person defaulted, 0 means they paid it back
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model, giving extra attention to defaults since they're rarer
model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Figure out the best cutoff point based on what mistakes actually cost us
# Rejecting a good borrower means we lose out on interest — about $500
# Approving a bad borrower means we lose the whole loan — about $5000
COST_FP = 500  
COST_FN = 5000 

y_probs = model.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0, 1, 100)
costs = []

# Try every threshold and calculate the total cost of mistakes at that cutoff
for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = (fp * COST_FP) + (fn * COST_FN)
    costs.append(total_cost)

# Pick the threshold that keeps our losses as low as possible
best_t = thresholds[np.argmin(costs)]

print(f"Optimal Threshold for Business: {best_t:.4f}")
print(f"Minimum Total Cost: ${min(costs):,}")

# Plot how cost changes across thresholds so we can see the sweet spot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, costs, color='blue', label='Total Business Cost')
plt.axvline(best_t, color='red', linestyle='--', label=f'Best Threshold ({best_t:.2f})')
plt.title('Business Cost Optimization')
plt.xlabel('Probability Threshold')
plt.ylabel('Cost ($)')
plt.legend()
plt.show()
