import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# List all step directories
step_dirs = sorted([d for d in os.listdir() if d.startswith('step_')])

# Feature names -- ORDER MUST MATCH FEATURE_SETS['portable_no_scores'] in
# final_train.incremental.py exactly:
# ['MAPQ', 'MISMATCHES', 'GAP_OPENS', 'GAP_EXT', 'EDIT_DIST',
#  'INSERT_SIZE', 'READ_GC_CONT', 'N_LOW_QUALITY_BASE', 'AVG_QUALITY_BASE_SCORE']
feature_names = [
    "MAPQ (Mapping Quality)",
    "MISMATCHES",
    "GAP_OPENS",
    "GAP_EXT (Gap Extensions)",
    "EDIT_DIST (Edit Distance)",
    "INSERT_SIZE",
    "READ_GC_CONT (Read GC Content)",
    "N_LOW_QUALITY_BASE",
    "AVG_QUALITY_BASE_SCORE"
]

# Initialize a dictionary to store feature importances
feature_importance_values = {name: [] for name in feature_names}

# Iterate over each step directory
for step_dir in step_dirs:
    model_path = os.path.join(step_dir, 'model.pkl')

    if os.path.exists(model_path):
        # Load the model
        xgb_model = joblib.load(model_path)

        # Extract feature importances
        importance = xgb_model.feature_importances_

        if len(importance) != len(feature_names):
            raise ValueError(
                f"{model_path}: model has {len(importance)} features but "
                f"feature_names has {len(feature_names)}. Wrong model for "
                f"this script (is this actually the portable_no_scores model?)."
            )

        # Append the importance values to the corresponding feature list
        for i, name in enumerate(feature_names):
            feature_importance_values[name].append(importance[i])

# Convert to DataFrame for plotting
importance_df = pd.DataFrame(feature_importance_values)

# Calculate median importance for sorting
median_importance = importance_df.median().sort_values(ascending=False)

# Sort the DataFrame by the median importance
sorted_importance_df = importance_df[median_importance.index]

flierprops = dict(marker='o', markerfacecolor='black', markersize=3, linestyle='none')
plt.figure(figsize=(6.5, 3.2))
sns.boxplot(data=sorted_importance_df, orient='h', linewidth=1, flierprops=flierprops)
plt.tight_layout()
plt.savefig('sorted_feature_importance_boxplot_portable.pdf', format='pdf', bbox_inches='tight')
plt.close()

print(f"Processed {len(feature_importance_values[feature_names[0]])} step models.")
print("Median importance ranking:")
print(median_importance)
