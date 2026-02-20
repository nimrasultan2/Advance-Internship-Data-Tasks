CREDIT RISK PREDICTOR

This project predicts whether a loan applicant will default using a LightGBM classifier trained on applicant and loan data. Missing values are filled with medians, categorical columns are one-hot encoded, and the data is split 80/20 for training and testing.
The highlight of the project is business cost optimization  instead of using a default 50% threshold, 
it tests 100 different cutoff points and picks the one that minimizes real financial loss, where approving a bad loan ($5,000 loss) is treated as far more costly than rejecting a good one ($500 loss).

1. Handles missing data and encodes categorical features before training
2.  Uses class_weight='balanced to handle the imbalance between defaults and non-defaults
3. Optimizes the decision threshold based on business cost, not just accuracy
4. Outputs the best threshold, minimum cost, and a cost curve plot
