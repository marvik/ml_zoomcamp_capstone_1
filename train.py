import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

# XGB Parameters 
xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 10,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
num_boost_round = 200
output_file = 'xgboost_model.bin'

# Data Loading and Preparation

# Load data 
df = pd.read_csv('online_shoppers_intention.csv')

# Data preprocessing
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Split the data
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Reset index
df_train_full = df_train_full.reset_index(drop=True)

# Separate features and target
X_train_full = df_train_full.drop('revenue', axis=1)
y_train_full = df_train_full.revenue.values

# Convert boolean columns to numerical (True -> 1, False -> 0)
bool_columns = X_train_full.select_dtypes(include='bool').columns
for col in bool_columns:
    X_train_full[col] = X_train_full[col].astype(int)

# Apply DictVectorizer
dv = DictVectorizer(sparse=False)
train_full_dict = X_train_full.to_dict(orient='records')
X_train_full = dv.fit_transform(train_full_dict)

# Train XGBoost model
features = dv.get_feature_names_out().tolist()
dfulltrain = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=features)

final_model = xgb.train(xgb_params, dfulltrain, num_boost_round=num_boost_round)

# Save the model and DictVectorizer
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)

print(f'The model is saved to {output_file}')