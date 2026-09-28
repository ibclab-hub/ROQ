import argparse
import os
import pickle

import numpy as np
import pandas as pd


def calculate_roq(rawP, k=10, maxMAPQ=42):
    return maxMAPQ * (1 / (1 + np.exp(-k * (rawP - 0.5))))


def corr_or_nan(x, y):
    # correlation is undefined (and numerically unstable) when one side
    # has ~zero variance, which happens near-universally for MAPQ once
    # you filter to a high-confidence subset (most retained reads sit at
    # the max MAPQ value) — report NaN rather than a misleading number
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


parser = argparse.ArgumentParser(description='Threshold sweep: MAPQ vs ROQ as read-filtering thresholds')
parser.add_argument('-m', '--model', help='path to model.pkl', required=True)
parser.add_argument('-i', '--input', help='feature+LABEL tsv (same format as training input)', required=True)
parser.add_argument('-head', '--head', help='header 1 for true, 0 for none', required=True)
parser.add_argument('-o', '--outdir', help='output directory', required=True)
parser.add_argument('--mapq-thresholds', help='comma-separated MAPQ thresholds (integers)', default='10,20,30,40')
parser.add_argument('--roq-thresholds', help='comma-separated ROQ thresholds (floats allowed, e.g. for fine-grained sweeps near saturation)', default='10,20,30,40')
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

os.makedirs(args.outdir, exist_ok=True)
mapq_thresholds = [float(t) for t in args.mapq_thresholds.split(',')]
roq_thresholds = [float(t) for t in args.roq_thresholds.split(',')]

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

MAPQ = df['MAPQ'].values
LABEL = df['LABEL'].values  # ROR ground truth

X = df[selected_features]

n_expected = getattr(model, 'n_features_in_', None)
if n_expected is not None and n_expected != X.shape[1]:
    raise ValueError(
        f"Model expects {n_expected} features but --feature-set '{args.feature_set}' "
        f"provides {X.shape[1]}. Did you pick the wrong --feature-set for this model?"
    )

rawROQ = model.predict(X.values)
ROQ = calculate_roq(rawROQ)

# ---- Table S3-style: fixed-threshold sweep ----
rows = []
for t in mapq_thresholds:
    mapq_mask = MAPQ >= t
    rows.append({
        'Metric': 'MAPQ', 'Threshold': t,
        'Retention': round(mapq_mask.mean() * 100, 2),
        'Mean_ROR': round(LABEL[mapq_mask].mean(), 4) if mapq_mask.any() else np.nan,
        'Corr_with_ROR': round(corr_or_nan(MAPQ[mapq_mask], LABEL[mapq_mask]), 3),
    })
for t in roq_thresholds:
    roq_mask = ROQ >= t
    rows.append({
        'Metric': 'ROQ', 'Threshold': t,
        'Retention': round(roq_mask.mean() * 100, 2),
        'Mean_ROR': round(LABEL[roq_mask].mean(), 4) if roq_mask.any() else np.nan,
        'Corr_with_ROR': round(corr_or_nan(ROQ[roq_mask], LABEL[roq_mask]), 3),
    })

fixed_df = pd.DataFrame(rows)
fixed_path = os.path.join(args.outdir, 'threshold_sweep_fixed.tsv')
fixed_df.to_csv(fixed_path, sep='\t', index=False)
print("=== Fixed-threshold sweep (Table S3 style) ===")
print(fixed_df.to_string(index=False))
print(f"Saved to {fixed_path}\n")

# ---- Figure S4-style: retention-matched comparison ----
# For each MAPQ threshold, find the ROQ threshold achieving the SAME
# retention rate (via quantile matching), then compare mean ROR at that
# matched retention level — isolates "does ROQ separate good/bad reads
# better than MAPQ, once you control for how many reads you keep".
matched_rows = []
for t in mapq_thresholds:
    mapq_mask = MAPQ >= t
    mapq_retention = mapq_mask.mean()
    if mapq_retention <= 0:
        continue

    # quantile matching: find q such that P(ROQ >= q) == mapq_retention
    matched_roq_threshold = np.quantile(ROQ, 1 - mapq_retention)
    roq_matched_mask = ROQ >= matched_roq_threshold

    matched_rows.append({
        'MAPQ_threshold': t,
        'Retention': round(mapq_retention * 100, 1),
        'MAPQ_mean_ROR': round(LABEL[mapq_mask].mean(), 4),
        'Matched_ROQ_threshold': round(matched_roq_threshold, 2),
        'ROQ_mean_ROR': round(LABEL[roq_matched_mask].mean(), 4),
    })

matched_df = pd.DataFrame(matched_rows)
matched_path = os.path.join(args.outdir, 'threshold_sweep_matched_retention.tsv')
matched_df.to_csv(matched_path, sep='\t', index=False)
print("=== Retention-matched comparison (Figure S4 style) ===")
print(matched_df.to_string(index=False))
print(f"Saved to {matched_path}")
