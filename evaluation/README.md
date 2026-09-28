# Reproducing the manuscript's results

This covers the scripts used to produce the manuscript's tables and
figures. You don't need any of this to just run ROQ on your own data --
see the top-level [`README.md`](../README.md) for that.

## Ground-truth generation and training

The production pipeline (`bin/ExtractFeatures.pl` -> `bin/predROQ.py`,
see the top-level README) is also how the training data itself was
built, with two additional ground-truth/training-only steps in between:

```
1. ExtractFeatures.pl <BAM|SAM|BED>              -> features.tsv
2. makeanswer_pysam.py -answer <sim SAM/BAM>     -> label.tsv
   -bed <out.bed from step 1>
   (reads ground truth directly from the simulator SAM/BAM rather than
   via an intermediate BED conversion)
3. paste features.tsv label.tsv > training_data.tsv
4. final_train.incremental.py --feature-set portable_no_scores
   (run once per data partition, warm-starting from the previous
   partition's model -- see the script's docstring for the
   incremental-boosting convention used to train model/model.pkl over
   120 partitions)
5. ../bin/predROQ.py --feature-set portable_no_scores  -> BAM tagged with ZQ
```

## Evaluation scripts

| Script | Produces |
|---|---|
| `hyperparam_search.py` + `model_comparison.py` | Regression-algorithm benchmark (Ridge/Lasso/DecisionTree/RandomForest/GradientBoosting/XGBoost) |
| `ablation_analysis.py` | Leave-one-feature-out ablation vs. the full 9-feature model |
| `feature_importance_dist_portable.py` | Feature-importance distribution across the 120 incremental training steps |
| `eval_on_new_data.py` | Cross-species / cross-aligner / cross-reference generalization |
| `threshold_sweep.py` | MAPQ- vs. ROQ-based filtering at fixed and retention-matched thresholds |
| `variant_calling_sweep_v3.py` + `summarize_vc_results.py` | GIAB variant-calling validation (filtering impact on precision/recall/F1) |
| `computational_cost_benchmark.sh` | Runtime, peak memory, and BAM-size overhead benchmark |

