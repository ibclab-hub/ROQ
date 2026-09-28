import argparse
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

parser = argparse.ArgumentParser(
    description='Ablation analysis: how much does each feature (and the full '
                'feature set) add over MAPQ alone.'
)
parser.add_argument('-i', '--input', help='feature summary tsv (same format as training input)', required=True)
parser.add_argument('-o', '--outdir', help='output directory', required=True)
parser.add_argument('-p', '--params', help='path to best_params.json from hyperparam_search.py '
                                            '(uses the XGBoost entry); optional, falls back to '
                                            'the defaults used in the main pipeline', required=False)
parser.add_argument('-n', '--n_repeats', help='number of random train/test splits to average over',
                     type=int, default=10)
parser.add_argument('--feature-set', choices=['full', 'portable_no_scores'], default='full',
                     help='full = all 13 features (original). portable_no_scores = drop '
                          'ALIGN_SCORE, SECONDARY_ALIGN_SCORE, MATE_ALIGNMENT_SCORE, '
                          'ALIGN_SCORE_DIFF -- run leave-one-out ablation within whichever '
                          'set the final model actually uses.')
args = parser.parse_args()

FEATURE_SETS = {
    'full': ['MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES', 'GAP_OPENS',
              'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
    'portable_no_scores': ['MAPQ', 'MISMATCHES', 'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
}

os.makedirs(args.outdir, exist_ok=True)

# same default XGBoost params used elsewhere in the pipeline; overridden by
# -p best_params.json if provided, so this stays consistent with whatever
# the final chosen hyperparameters turn out to be
xgb_params = {
    'colsample_bytree': 0.6,
    'learning_rate': 0.1,
    'max_depth': 7,
    'min_child_weight': 5,
    'n_estimators': 200,
    'subsample': 1.0,
}
if args.params:
    with open(args.params) as f:
        best = json.load(f)
    if 'XGBoost' in best:
        xgb_params.update(best['XGBoost']['best_params'])
        print(f"Using tuned XGBoost params from {args.params}: {xgb_params}")

df = pd.read_csv(args.input, sep='\t')
df = df.drop(['READ_NAME'], axis=1)

X_full = df[FEATURE_SETS[args.feature_set]]
y = df['LABEL']
feature_names = list(X_full.columns)

if 'MAPQ' not in feature_names:
    raise ValueError("Expected a 'MAPQ' column in the input.")


def fit_eval(X_train, X_test, y_train, y_test, seed):
    model = XGBRegressor(**xgb_params, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R^2': r2_score(y_test, y_pred),
        'pred_corr': np.corrcoef(y_test, y_pred)[0, 1],
    }


rows = []

for repeat in range(args.n_repeats):
    seed = repeat  # varies the split each repeat, for robustness across samples

    # ---- variant 1: MAPQ only ----
    X_mapq = X_full[['MAPQ']]
    X_tr, X_te, y_tr, y_te = train_test_split(X_mapq, y, test_size=0.2, random_state=seed)
    metrics = fit_eval(X_tr, X_te, y_tr, y_te, seed)
    rows.append({'variant': 'mapq_only', 'repeat': repeat, **metrics})

    # ---- variant 2: full feature set ----
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=seed)
    metrics = fit_eval(X_tr, X_te, y_tr, y_te, seed)
    rows.append({'variant': 'full', 'repeat': repeat, **metrics})

    # ---- variant 3: leave-one-feature-out, for every non-MAPQ feature ----
    # (dropping MAPQ itself is a separate, arguably more interesting ablation;
    # included here as its own row so it's easy to compare against mapq_only)
    for feat in feature_names:
        X_loo = X_full.drop(columns=[feat])
        X_tr, X_te, y_tr, y_te = train_test_split(X_loo, y, test_size=0.2, random_state=seed)
        metrics = fit_eval(X_tr, X_te, y_tr, y_te, seed)
        rows.append({'variant': f'without_{feat}', 'repeat': repeat, **metrics})

    print(f"repeat {repeat+1}/{args.n_repeats} done")

results_df = pd.DataFrame(rows)
raw_path = os.path.join(args.outdir, 'ablation_raw.tsv')
results_df.to_csv(raw_path, sep='\t', index=False)

# summary: mean/std per variant, plus delta vs the full model (how much
# performance is lost by removing that feature / by using MAPQ alone)
summary = results_df.groupby('variant')[['MSE', 'RMSE', 'MAE', 'R^2', 'pred_corr']].agg(['mean', 'std'])
summary.columns = ['_'.join(c) for c in summary.columns]
summary = summary.reset_index()

full_r2_mean = summary.loc[summary['variant'] == 'full', 'R^2_mean'].values[0]
full_mse_mean = summary.loc[summary['variant'] == 'full', 'MSE_mean'].values[0]
summary['delta_R2_vs_full'] = summary['R^2_mean'] - full_r2_mean
summary['delta_MSE_vs_full'] = summary['MSE_mean'] - full_mse_mean

summary = summary.sort_values('delta_R2_vs_full')

summary_path = os.path.join(args.outdir, 'ablation_summary.tsv')
summary.to_csv(summary_path, sep='\t', index=False)

print(f"\nSaved per-repeat results to {raw_path}")
print(f"Saved summary (mean/std, delta vs full model) to {summary_path}")
print("\nFeatures whose removal hurts R^2 the most (most negative delta) are the most important:")
print(summary[['variant', 'R^2_mean', 'delta_R2_vs_full']].to_string(index=False))
