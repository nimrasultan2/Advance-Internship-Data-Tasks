TERM DEPOSIT SUBSCRIPTION PREDICTION 

Loaded dataset and basic formatting to make the dataset ready to use.
It predicts whether a bank customer will subscribe to a term deposit using the Bank Marketing dataset. 
The objective is to support data-driven marketing decisions by identifying customers with a higher likelihood of subscription.
The dataset was cleaned and preprocessed using pandas, including handling categorical variables through one-hot encoding and converting the target variable into binary format. 
The data was split into training and testing sets with stratification to preserve class balance. 
Logistic Regression and Random Forest Classifier were implemented as baseline and advanced models respectively. Feature scaling was applied where necessary.
Model performance was evaluated using F1-score, ROC-AUC, confusion matrix, and classification report. 
The Random Forest model demonstrated stronger performance due to its ability to capture non-linear patterns and feature interactions. 
To enhance interpretability, SHAP (SHapley Additive Explanations) was used to explain individual predictions and identify key features influencing customer subscription behavior.
