import argparse
import pickle
import os
import sys
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import pysam

# Utility functions
def calculate_roq(rawP, k=10, maxMAPQ=42):
    # maxMAPQ=42 to match the scale used throughout the rest of the
    # pipeline (training, threshold_sweep.py, ablation) -- previously this
    # was 60 here only, which silently produced ROQ values on a different
    # scale than everywhere else.
    return maxMAPQ * (1 / (1 + np.exp(-k * (rawP - 0.5))))

# The SAM tag we write ROQ into. Deliberately NOT "RQ": PacBio HiFi/CCS BAMs
# already carry a native "RQ:f:" tag (predicted read quality from CCS):
# reusing that code would silently overwrite/collide with it. "Z*" tags are
# reserved by the SAM spec for local/custom use and are safe.
# NOTE: if you change this, also update: the manuscript (Methods /
# Computational cost section, wherever "RQ tag" is mentioned), any
# GitHub/Zenodo README usage examples, and downstream scripts that read
# this tag back out of the BAM.
ROQ_TAG = "ZQ"

# Argument parser
parser = argparse.ArgumentParser(description='Add ROQ tags to BAM file using a trained model.')
parser.add_argument('-i', '--input', help='Feature summary file (tab-delimited)', required=True)
parser.add_argument('-ib', '--inputBam', help='Input original BAM file', required=True)
parser.add_argument('-o', '--outdir', help='Output directory for results', required=True)
parser.add_argument('-m', '--model', help='Trained model to use', required=False)
parser.add_argument('-head', '--head', help='header 1 for true, 0 for none (matches final_train.incremental.py)', required=True)
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

# Paths and directories
input_bam = args.inputBam
feature_file = args.input
outdir = args.outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)

output_bam = os.path.join(outdir, 'output.bam')

# Load the trained model
if not args.model:
    sys.exit("Error: -m/--model is required (no pretrained model was given, "
              "and an untrained XGBRegressor cannot produce predictions).")

with open(args.model, 'rb') as file:
    xgb_model = pickle.load(file)

# Load the feature file -- same column set/order as the training pipeline
# (final_train.incremental.py), minus LABEL, since production data has no
# ground truth to predict against.
if int(args.head) == 1:
    df = pd.read_csv(feature_file, sep='\t')
else:
    df = pd.read_csv(feature_file, sep='\t', header=None)
    df.columns = [
        'READ_NAME', 'MAPQ', 'ALIGN_SCORE', 'SECONDARY_ALIGN_SCORE', 'MISMATCHES',
        'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST', 'MATE_ALIGNMENT_SCORE', 'ALIGN_SCORE_DIFF',
        'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE']

# No scaling/imputation: the current model was trained on RAW features
# (no SimpleImputer/MinMaxScaler in the training pipeline -- XGBoost
# handles NaN natively, and tree splits are invariant to monotonic
# scaling). Applying a freshly-fit scaler/imputer here, as an earlier
# version of this script did, would feed the model numbers on a totally
# different scale than it was trained on and silently produce garbage
# predictions. Missing values (NA in the feature file) are passed through
# as NaN, which XGBoost handles directly.
X = df[selected_features].apply(pd.to_numeric, errors='coerce')

n_expected = getattr(xgb_model, 'n_features_in_', None)
if n_expected is not None and n_expected != X.shape[1]:
    raise ValueError(
        f"Model expects {n_expected} features but --feature-set '{args.feature_set}' "
        f"provides {X.shape[1]}. Did you pick the wrong --feature-set for this model?"
    )

y_pred = xgb_model.predict(X.values)

# Process BAM file
samfile = pysam.AlignmentFile(input_bam, 'r')
header = samfile.header.to_dict()

# Add ROQ tag to header
header.setdefault('PG', []).append({
    'ID': 'ROQ',
    'PN': 'ROQ Tag',
    'CL': 'Read Overlapping Quality',
    'VN': '1.0'
})

outfile = pysam.AlignmentFile(output_bam, 'wb', header=header)

# Add ROQ to each read
#
# IMPORTANT: paired-end reads share the same READ_NAME (QNAME), and the
# feature file has one row per alignment record (i.e. one row per mate /
# per multi-mapping record), in the same order they were read from the
# BAM/BED file that -i was derived from. This FIFO-queue-per-name approach
# only assigns the correct ROQ to the correct read if that row order
# EXACTLY matches the order reads are encountered when iterating -ib with
# pysam below. If -i was generated from a differently-sorted or
# independently-derived BED/BAM, mate1 could silently receive mate2's ROQ
# value (or vice versa). The count check below catches gross mismatches
# but not a same-length reordering -- if in doubt, regenerate -i directly
# from -ib via the same conversion path (no separate re-sort in between).
roq_values = calculate_roq(y_pred)
read_name_queue = defaultdict(deque)
for name, roq in zip(df['READ_NAME'], roq_values):
    read_name_queue[name].append(roq)

total_feature_rows = len(df)
total_bam_reads = sum(1 for _ in pysam.AlignmentFile(input_bam, 'r'))
if total_feature_rows != total_bam_reads:
    print(f"WARNING: feature file has {total_feature_rows} rows but {input_bam} "
          f"has {total_bam_reads} reads. These should normally match exactly "
          f"(same BAM -> same conversion -> feature extraction, no reads "
          f"dropped or added in between). Mismatched counts are a strong "
          f"signal that -i and -ib are not in the same read order, which "
          f"can cause reads to be silently tagged with the wrong ROQ value.",
          file=sys.stderr)

unmatched_reads = 0
for read in samfile:
    queue = read_name_queue.get(read.query_name)
    if queue:
        roq_value = queue.popleft()
        read.set_tag(ROQ_TAG, float(roq_value), value_type='f')
    else:
        unmatched_reads += 1
    outfile.write(read)

if unmatched_reads:
    print(f"Warning: {unmatched_reads} read(s) in {input_bam} had no matching "
          f"row in {feature_file} and were written without a {ROQ_TAG} tag.",
          file=sys.stderr)

# leftover, un-consumed feature rows indicate feature_file had MORE rows for
# some READ_NAME than -ib had actual reads -- also a sign of a mismatch
leftover = sum(len(q) for q in read_name_queue.values())
if leftover:
    print(f"WARNING: {leftover} feature-file row(s) were never consumed "
          f"(more rows for some READ_NAME than reads found in {input_bam}). "
          f"This points at a mismatch between -i and -ib.",
          file=sys.stderr)

samfile.close()
outfile.close()
print(f"Updated Alignment file saved to {output_bam} (tag: {ROQ_TAG})")
