import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_roq(rawP, k=10, maxMAPQ=42):
    return maxMAPQ * (1 / (1 + np.exp(-k * (rawP - 0.5))))


parser = argparse.ArgumentParser(
    description='Evaluate a trained model on new, never-before-seen raw feature data '
                '(same column format as the training input: READ_NAME, MAPQ, ..., LABEL).'
)
parser.add_argument('-m', '--model', help='path to model.pkl', required=True)
parser.add_argument('-i', '--input', help='new raw feature tsv (e.g. a held-out simulation run)', required=True)
parser.add_argument('-head', '--head', help='header 1 for true, 0 for none', required=True)
parser.add_argument('-model_name', '--model_name', help='label for this row, e.g. sequential / shuffled', required=True)
parser.add_argument('-o', '--out', help='path to the comparison tsv to append metrics to', default='model_compare.tsv')
parser.add_argument('-roq_out', '--roq_out', help='optional path to save per-read predictions '
                                                    '(READ_NAME, LABEL, rawROQ, ROQ) for downstream figures',
                     required=False)
parser.add_argument('--feature-set', choices=['full', 'portable_no_scores'], default='full',
                     help='must match the feature set the model was trained with')
args = parser.parse_args()

FEATURE_SETS = {
    'full': ['MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES', 'GAP_OPENS',
              'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
    'portable_no_scores': ['MAPQ', 'MISMATCHES', 'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST',
              'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE'],
}
selected_features = FEATURE_SETS[args.feature_set]

with open(args.model, 'rb') as f:
    model = pickle.load(f)

if int(args.head) == 1:
    df = pd.read_csv(args.input, sep='\t')
else:
    df = pd.read_csv(args.input, sep='\t', header=None)
    df.columns = [
        'READ_NAME', 'MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES',
        'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
        'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE', 'LABEL']

read_names = df['READ_NAME']
X = df[selected_features]
y = df['LABEL']

n_expected = getattr(model, 'n_features_in_', None)
if n_expected is not None and n_expected != X.shape[1]:
    raise ValueError(
        f"Model expects {n_expected} features but --feature-set '{args.feature_set}' "
        f"provides {X.shape[1]}. Did you pick the wrong --feature-set for this model?"
    )

y_pred = model.predict(X.values)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
pred_corr = np.corrcoef(y, y_pred)[0, 1]
n_rounds = model.get_booster().num_boosted_rounds()

row = pd.DataFrame([{
    'model_name': args.model_name,
    'n_boosted_rounds_total': n_rounds,
    'n_rows_evaluated': len(df),
    'MSE': round(mse, 4),
    'RMSE': round(rmse, 4),
    'MAE': round(mae, 4),
    'R^2': round(r2, 4),
    'pred_corr': round(pred_corr, 4),
}])

if os.path.exists(args.out):
    row.to_csv(args.out, sep='\t', mode='a', header=False, index=False)
else:
    row.to_csv(args.out, sep='\t', mode='w', header=True, index=False)

print(f"[{args.model_name}] n_rounds={n_rounds}  R^2={r2:.4f}  RMSE={rmse:.4f}  pred_corr={pred_corr:.4f}")
print(f"Appended to {args.out}")

if args.roq_out:
    roq = calculate_roq(y_pred)
    out_df = pd.DataFrame({
        'READ_NAME': read_names,
        'LABEL': y,
        'rawROQ': y_pred,
        'ROQ': roq,
    })
    out_df.to_csv(args.roq_out, sep='\t', index=False)
    print(f"Saved per-read predictions to {args.roq_out}")
