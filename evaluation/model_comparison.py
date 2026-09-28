import argparse
import json
import pickle
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

FEATURE_SETS = {
    'full': ['MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES', 'GAP_OPENS',
              'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
    'portable_no_scores': ['MAPQ', 'MISMATCHES', 'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
}

##
def calculate_evaluation_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
    return mse, rmse, mae, r2, mape

##

parser = argparse.ArgumentParser(description='ML model comparison for MAPQ correction (baseline selection, no ROQ here)')
parser.add_argument('-i', '--input', help='Input bam file feature summary data', required=True)
parser.add_argument('-o', '--outdir', help='output directory saving trained models + comparison table', required=True)
parser.add_argument('--feature-set', choices=['full', 'portable_no_scores'], default='full',
                     help='must match the feature set the final production model uses')
parser.add_argument('-p', '--params', help='path to best_params.json from hyperparam_search.py '
                                            '(same run/feature-set); if omitted, falls back to '
                                            'the hardcoded defaults below', required=False)
args = parser.parse_args()

file_path = args.input
outDir_path = args.outdir
if not os.path.exists(outDir_path):
    os.makedirs(outDir_path, exist_ok=True)

df = pd.read_csv(file_path, sep='\t')
df = df.drop(['READ_NAME'], axis=1)

X = df[FEATURE_SETS[args.feature_set]]
y = df['LABEL']
column_info = X.columns

# split FIRST, fit imputer/scaler on train only — avoids leaking test
# statistics into preprocessing
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train_raw)
X_test_imp = imputer.transform(X_test_raw)

def linear_pipeline(model):
    return Pipeline([('scaler', StandardScaler()), ('model', model)])

# Fallback hyperparameters (from the original full-feature search).
# These get overridden by -p best_params.json when provided, so re-running
# hyperparam_search.py with --feature-set portable_no_scores and pointing
# -p at its output keeps this comparison apples-to-apples with whichever
# feature set the production model actually uses.
default_params = {
    'RidgeRegression': {'alpha': 1.0000186554302904e-05},
    'LassoRegression': {'alpha': 0.00011984710102768185},
    'DecisionTreeRegressor': {'max_depth': 5},
    'RandomForestRegressor': {'n_estimators': 91, 'max_depth': 7},
    'GradientBoostingRegressor': {'n_estimators': 13, 'learning_rate': 0.26865393138361204, 'max_depth': 5},
    'XGBoostRegressor': {'n_estimators': 74, 'max_depth': 6, 'learning_rate': 0.049464249689796146},
}

if args.params:
    with open(args.params) as f:
        tuned = json.load(f)
    # hyperparam_search.py keys are like 'XGBoost', 'RandomForest', etc.
    # (no 'Regressor' suffix) — map onto the *Regressor names used here
    name_map = {
        'XGBoost': 'XGBoostRegressor',
        'GradientBoosting': 'GradientBoostingRegressor',
        'RandomForest': 'RandomForestRegressor',
        'DecisionTree': 'DecisionTreeRegressor',
        'Lasso': 'LassoRegression',
        'Ridge': 'RidgeRegression',
    }
    for search_name, entry in tuned.items():
        model_name = name_map.get(search_name)
        if model_name in default_params:
            default_params[model_name].update(entry['best_params'])
            print(f"Using tuned params for {model_name}: {default_params[model_name]}")

p = default_params

models = [
    ('LinearRegression', linear_pipeline(LinearRegression())),
    ('RidgeRegression', linear_pipeline(Ridge(random_state=RANDOM_STATE, **p['RidgeRegression']))),
    ('LassoRegression', linear_pipeline(Lasso(random_state=RANDOM_STATE, **p['LassoRegression']))),
    ('SGDRegressor', linear_pipeline(SGDRegressor(random_state=RANDOM_STATE))),
    ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=RANDOM_STATE, **p['DecisionTreeRegressor'])),
    ('RandomForestRegressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **p['RandomForestRegressor'])),
    ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=RANDOM_STATE, **p['GradientBoostingRegressor'])),
    ('XGBoostRegressor', XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, **p['XGBoostRegressor'])),
]

coef_models = {"LinearRegression", "RidgeRegression", "LassoRegression", "SGDRegressor"}

MAPQ = X['MAPQ'].values
LABEL = y.values
corr_MAPQ_LABEL = np.corrcoef(MAPQ, LABEL)[0, 1]
print(f"init_corr (MAPQ vs LABEL, full data): {corr_MAPQ_LABEL:.4f}")

results = []
importances = {}

for name, model in models:
    model.fit(X_train_imp, y_train)
    y_pred = model.predict(X_test_imp)

    model_save_fName = os.path.join(outDir_path, f"{name}_model.pkl")
    with open(model_save_fName, 'wb') as file:
        pickle.dump(model, file)

    if name in coef_models:
        importances[name] = dict(zip(column_info, model.named_steps['model'].coef_))
    else:
        importances[name] = dict(zip(column_info, model.feature_importances_))

    mse, rmse, mae, r2, mape = calculate_evaluation_metrics(y_test, y_pred)
    pred_corr = np.corrcoef(y_test, y_pred)[0, 1]

    results.append({
        'Machine learning algorithms': name,
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R^2': round(r2, 4),
        'Correlation (r)': round(pred_corr, 4),
    })
    print(f"{name}  MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R^2: {r2:.4f}  corr: {pred_corr:.4f}")

results_df = pd.DataFrame(results)
results_table_path = os.path.join(outDir_path, 'model_comparison.tsv')
results_df.to_csv(results_table_path, sep='\t', index=False)
print(f"\nSaved comparison table to {results_table_path}")

importance_df = pd.DataFrame(importances).T
importance_path = os.path.join(outDir_path, 'feature_importance.tsv')
importance_df.to_csv(importance_path, sep='\t')
print(f"Saved feature importances/coefficients to {importance_path}")
