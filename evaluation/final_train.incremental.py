import argparse
import pickle
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

##
def calculate_evaluation_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
    return mse, rmse, mae, r2, mape

##
def calculate_log10(p, e=1e-6):
    return -np.log10(1 - p + e)
##
def calculate_roq(rawP, k=10, maxMAPQ=42):
    roq = maxMAPQ * (1 / (1 + np.exp(-k * (rawP - 0.5))))
    return roq

parser = argparse.ArgumentParser(description='MAPQ Correction (incremental, no scaler/imputer)')
parser.add_argument('-i', '--input', help='Input bam file feature summary data', required=True)
parser.add_argument('-o', '--outdir', help='output directory saving trained model', required=True)
parser.add_argument('-m', '--model', help='trained model to be updated (path to previous step .pkl)', required=False)
parser.add_argument('-s', '--save', help='save model name', required=True)
parser.add_argument('-head', '--head', help='header 1 for true 0 for none', required=True)
parser.add_argument('--feature-set', choices=['full', 'portable_no_scores'], default='full',
                     help='full = all 13 features (original). portable_no_scores = drop '
                          'ALIGN_SCORE, SECONDARY_ALIGN_SCORE, MATE_ALIGNMENT_SCORE, '
                          'ALIGN_SCORE_DIFF (the aligner-score features that do not '
                          'transfer across aligners with different scoring conventions).')
args = parser.parse_args()

FEATURE_SETS = {
    'full': ['MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES', 'GAP_OPENS',
              'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
    'portable_no_scores': ['MAPQ', 'MISMATCHES', 'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
}
selected_features = FEATURE_SETS[args.feature_set]

file_path = args.input
outDir_path = args.outdir
name = args.save

if not os.path.exists(outDir_path):
    os.makedirs(outDir_path, exist_ok=True)

best_params = {
    'colsample_bytree': 0.6,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 5,
    'n_estimators': 200,
    'subsample': 1.0
}

xgb_model = XGBRegressor(**best_params)

prev_booster = None
if args.model:
    model_path = args.model
    print(f"Updating model from {model_path}")
    with open(model_path, 'rb') as file:
        xgb_model = pickle.load(file)
    # continue boosting from the previous round instead of starting over
    prev_booster = xgb_model.get_booster()
else:
    print("Creating a new model (step 0)")

df = pd.DataFrame()
if int(args.head) == 1:
    df = pd.read_csv(file_path, sep='\t')
else:
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = [
        'READ_NAME', 'MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES',
        'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
        'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE', 'LABEL']

data = df.to_dict(orient='list')
df = pd.DataFrame(data)
df = df.drop(['READ_NAME'], axis=1)

X = df[selected_features]
y = df['LABEL']
df = X
column_info = df.columns

# No imputer / scaler:
# - XGBoost handles NaN natively and learns the best split direction for
#   missing values as part of training, which tends to work better than
#   mean-imputation for tree models.
# - Tree splits are invariant to monotonic transforms like MinMaxScaler,
#   so scaling raw features doesn't change what the model can learn — it
#   only risked inconsistent normalization across chunks in the old
#   pipeline. Removing it also removes that whole failure mode.
X_raw = X.values

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
sample_weights_train = np.where(y_train > 0.75, 2, 1)

mapq_idx = list(column_info).index('MAPQ')
MAPQ = df['MAPQ'].values
LABEL = y
corr_MAPQ_LABEL = np.corrcoef(MAPQ, LABEL)[0, 1]
print(f"init_all_corr: {corr_MAPQ_LABEL:.4f}")

# true incremental boosting: continue from prev_booster if present
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    xgb_model=prev_booster
)

y_pred = xgb_model.predict(X_test)
model_save_fName = os.path.join(outDir_path, name + ".pkl")

with open(model_save_fName, 'wb') as file:
    pickle.dump(xgb_model, file)

mse, rmse, mae, r2, mape = calculate_evaluation_metrics(y_test, y_pred)
pred_corr = np.corrcoef(y_test, y_pred)[0, 1]

roq = calculate_roq(y_pred)
df_final = pd.DataFrame(X_test, columns=column_info)
df_final['LABEL'] = y_test.values
df_final['rawROQ'] = y_pred
df_final['ROQ'] = roq

corr_MAPQ_LABEL = np.corrcoef(X_test[:, mapq_idx], y_test)[0, 1]
print(f"init_test_corr: {corr_MAPQ_LABEL:.4f}")

output_file = os.path.join(outDir_path, 'roq.tsv')
df_final.to_csv(output_file, sep='\t', index=False)

print(f"{name}  n_rows_this_chunk: {len(df)}")
print(f"{name}  n_boosted_rounds_total: {xgb_model.get_booster().num_boosted_rounds()}")
print(f"{name}  MSE: {mse:.4f}")
print(f"{name}  RMSE: {rmse:.4f}")
print(f"{name}  MAE: {mae:.4f}")
print(f"{name}  R^2 Score: {r2:.4f}")
print(f"{name}  pred_corr: {pred_corr:.4f}")
