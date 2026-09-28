import argparse
import json
import os

import optuna
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42

parser = argparse.ArgumentParser(description='MAPQ Correction - hyperparameter search (train-only, CV)')
parser.add_argument('-i', '--input', help='Input bam file feature summary data', required=True)
parser.add_argument('-o', '--outdir', help='output dir for best_params.json', required=True)
parser.add_argument('--feature-set', choices=['full', 'portable_no_scores'], default='full',
                     help='must match the feature set the final production model uses, '
                          'so this comparison is apples-to-apples')
parser.add_argument('--n-trials', type=int, default=150,
                     help='Optuna trials per model. 1000 gives diminishing returns for '
                          '2-3 hyperparameters; 100-150 is generally enough for TPE to '
                          'converge close to optimal for this search space size.')
parser.add_argument('--n-splits', type=int, default=3,
                     help='CV folds. 3 instead of 5 cuts fit count by 40% with only a small '
                          'increase in variance for this purpose (picking hyperparameters, '
                          'not reporting final numbers).')
parser.add_argument('--sample-n', type=int, default=500000,
                     help='Subsample this many rows from the training split before CV search. '
                          'The slow models here (GradientBoostingRegressor especially, no '
                          'internal parallelism; RandomForestRegressor with deep trees) scale '
                          'poorly with row count. Hyperparameters found on a representative '
                          '500K-row subsample transfer well to the full data. Set to 0 to '
                          'disable and use the full training split (slow).')
args = parser.parse_args()

N_TRIALS = args.n_trials
N_SPLITS = args.n_splits

FEATURE_SETS = {
    'full': ['MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES', 'GAP_OPENS',
              'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
    'portable_no_scores': ['MAPQ', 'MISMATCHES', 'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
}

file_path = args.input
outDir_path = args.outdir
if not os.path.exists(outDir_path):
    os.makedirs(outDir_path, exist_ok=True)

df = pd.read_csv(file_path, sep='\t')
df = df.drop(['READ_NAME'], axis=1)

X = df[FEATURE_SETS[args.feature_set]]
y = df['LABEL']

# Hold out the test set FIRST and never touch it again in this script.
# It only gets used once, later, in the final comparison/report script.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

if args.sample_n and len(X_train_raw) > args.sample_n:
    sample_idx = X_train_raw.sample(n=args.sample_n, random_state=RANDOM_STATE).index
    X_train_raw = X_train_raw.loc[sample_idx]
    y_train = y_train.loc[sample_idx]
    print(f"Subsampled training split to {args.sample_n} rows for the search "
          f"(full training split had more rows; final params still get validated "
          f"on the full held-out test set downstream).")

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train_raw)
# X_test intentionally not transformed/used here — search must not see it

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def cv_mse(model, X, y):
    # cross_val_score maximizes by convention, so neg-MSE; flip sign back
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()


def optimize_ridge(trial):
    alpha = trial.suggest_float('alpha', 1e-5, 10, log=True)
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    return cv_mse(model, X_train, y_train)


def optimize_lasso(trial):
    alpha = trial.suggest_float('alpha', 0.0001, 10.0, log=True)
    model = Lasso(alpha=alpha, random_state=RANDOM_STATE)
    return cv_mse(model, X_train, y_train)


def optimize_decision_tree(trial):
    max_depth = trial.suggest_int('max_depth', 1, 32)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE)
    return cv_mse(model, X_train, y_train)


def optimize_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    return cv_mse(model, X_train, y_train)


def optimize_gradient_boosting(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, random_state=RANDOM_STATE
    )
    return cv_mse(model, X_train, y_train)


def optimize_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 32)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=RANDOM_STATE
    )
    return cv_mse(model, X_train, y_train)


searches = [
    ('XGBoost', optimize_xgboost),
    ('GradientBoosting', optimize_gradient_boosting),
    ('RandomForest', optimize_random_forest),
    ('DecisionTree', optimize_decision_tree),
    ('Lasso', optimize_lasso),
    ('Ridge', optimize_ridge),
]

best_params_all = {}
for name, objective_fn in searches:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_fn, n_trials=N_TRIALS)
    print(f'[RES] {name} Best Params (train-CV only):', study.best_params)
    best_params_all[name] = {
        'best_params': study.best_params,
        'best_cv_mse': study.best_value,
    }

out_path = os.path.join(outDir_path, 'best_params.json')
with open(out_path, 'w') as f:
    json.dump(best_params_all, f, indent=2)

print(f"\nSaved best params (never evaluated against test set) to {out_path}")
print("Next step: plug these into the comparison script, which fits ONCE on "
      "full train data with these params and evaluates ONCE on the held-out test set.")
