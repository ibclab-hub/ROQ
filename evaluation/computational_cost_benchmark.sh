#!/bin/bash
# Computational cost benchmark (runtime / peak memory / BAM-size overhead)
#
# Usage:
#   ./computational_cost_benchmark.sh <full.bam> [script_dir] [model_path] [outdir]
#
#   full.bam    A BAM with at least 1M mapped reads to subsample from.
#   script_dir  Directory containing ExtractFeatures.pl and predROQ.py
#               (default: ../bin, i.e. this repo's bin/ directory).
#   model_path  Path to model.pkl (default: ../model/model.pkl).
#   outdir      Where to write results (default: ./cost_benchmark).
#
# Requires `/usr/bin/time -v` (GNU coreutils time, not bash's builtin `time`)
# for peak memory reporting. If missing: `sudo yum install time` / `apt install time`.

set -euo pipefail

FULL_BAM=$1
SCRIPT_DIR=${2:-"$(dirname "$0")/../bin"}
MODEL=${3:-"$(dirname "$0")/../model/model.pkl"}
OUTDIR=${4:-./cost_benchmark}
mkdir -p "$OUTDIR"

RESULTS="$OUTDIR/computational_cost.tsv"
echo -e "Reads\tFeature_time_s\tPredict_time_s\tTotal_time_s\tPeak_mem_MB\tBAM_size_MB\tROQ_BAM_size_MB\tOverhead_pct" > "$RESULTS"

for N in 10000 100000 1000000; do
    echo "=== Benchmarking N=$N reads ==="
    SUBDIR="$OUTDIR/n${N}"
    mkdir -p "$SUBDIR"
    SUB_BAM="$SUBDIR/subsample.bam"
    FEATURES="$SUBDIR/features.tsv"
    ROQ_BAM="$SUBDIR/subsample.roq.bam"

    # subsample N reads (deterministic: first N mapped reads)
    samtools view -h "$FULL_BAM" | head -n $((N + 5)) | samtools view -Sb - > "$SUB_BAM" 2>/dev/null
    samtools index "$SUB_BAM" 2>/dev/null || true

    ACTUAL_READS=$(samtools view -c "$SUB_BAM")
    echo "  actual reads in subsample: $ACTUAL_READS"

    # --- feature extraction timing + memory ---
    FEATURE_TIME_FILE="$SUBDIR/feature_time.txt"
    /usr/bin/time -v perl "${SCRIPT_DIR}/ExtractFeatures.pl" "$SUB_BAM" > "$FEATURES" 2> "$FEATURE_TIME_FILE"
    FEATURE_TIME=$(grep "Elapsed (wall clock)" "$FEATURE_TIME_FILE" | awk -F': ' '{print $2}')
    FEATURE_MEM_KB=$(grep "Maximum resident set size" "$FEATURE_TIME_FILE" | awk -F': ' '{print $2}')

    # --- predROQ.py timing + memory ---
    PREDICT_TIME_FILE="$SUBDIR/predict_time.txt"
    /usr/bin/time -v python3 "${SCRIPT_DIR}/predROQ.py" \
        --feature-set portable_no_scores \
        -m "$MODEL" \
        -head 0 \
        -i "$FEATURES" \
        -ib "$SUB_BAM" \
        -o "$ROQ_BAM" 2> "$PREDICT_TIME_FILE"
    PREDICT_TIME=$(grep "Elapsed (wall clock)" "$PREDICT_TIME_FILE" | awk -F': ' '{print $2}')
    PREDICT_MEM_KB=$(grep "Maximum resident set size" "$PREDICT_TIME_FILE" | awk -F': ' '{print $2}')

    # --- file sizes ---
    BAM_SIZE_MB=$(du -m "$SUB_BAM" | cut -f1)
    ROQ_BAM_SIZE_MB=$(du -m "$ROQ_BAM/output.bam" 2>/dev/null | cut -f1 || echo "NA")
    if [[ "$ROQ_BAM_SIZE_MB" != "NA" && "$BAM_SIZE_MB" -gt 0 ]]; then
        OVERHEAD=$(python3 -c "print(round(($ROQ_BAM_SIZE_MB - $BAM_SIZE_MB) / $BAM_SIZE_MB * 100, 1))")
    else
        OVERHEAD="NA"
    fi

    # peak memory across both stages (max of the two)
    PEAK_MEM_KB=$(( FEATURE_MEM_KB > PREDICT_MEM_KB ? FEATURE_MEM_KB : PREDICT_MEM_KB ))
    PEAK_MEM_MB=$(( PEAK_MEM_KB / 1024 ))

    echo -e "${ACTUAL_READS}\t${FEATURE_TIME}\t${PREDICT_TIME}\t-\t${PEAK_MEM_MB}\t${BAM_SIZE_MB}\t${ROQ_BAM_SIZE_MB}\t${OVERHEAD}" >> "$RESULTS"

    echo "  feature_time=${FEATURE_TIME}  predict_time=${PREDICT_TIME}  peak_mem=${PEAK_MEM_MB}MB  overhead=${OVERHEAD}%"
done

echo ""
echo "Done. Results:"
cat "$RESULTS"
