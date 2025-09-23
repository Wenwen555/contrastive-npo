import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Hyperparameter values
betas = [0.05, 0.1, 0.2, 0.5]
neg_samples = [2, 3, 4]

# Data
verb_mem = np.array([
    [0.65, 1.87, 2.67],
    [3.20, 3.68, 3.85],
    [1.85, 2.08, 0.59],
    [2.02, 1.73, 2.19],
])

model_utility = np.array([
    [0.28, 0.69, 0.79],
    [0.36, 0.73, 0.64],
    [0.64, 0.69, 0.65],
    [0.60, 0.56, 0.61],
])

# Transpose and create DataFrames
df_verb_mem = pd.DataFrame(verb_mem.T, index=neg_samples, columns=betas)
df_model_utility = pd.DataFrame(model_utility.T, index=neg_samples, columns=betas)

# Plot heatmaps side by side
plt.figure(figsize=(14, 6))

# sns.heatmap(df_verb_mem, annot=True, fmt=".2f", cmap="YlGnBu_r", cbar=False, square=True,
#             linewidths=4, linecolor='white', annot_kws={"fontsize": 25})
# Heatmap: model_utility (transposed)
sns.heatmap(df_model_utility, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False, square=True,
            linewidths=4, linecolor='white', annot_kws={"fontsize": 25})

plt.xlabel(r"$\beta$", fontsize=29)
plt.ylabel(r"$k$", fontsize=29)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.yticks(rotation=0)


plt.tight_layout()
# plt.savefig("/data/home/jvnting/cnpo/plot/heatmap_muse_verbmem.pdf", dpi=300)
plt.savefig("/data/home/jvnting/cnpo/plot/heatmap_muse_model_utility.pdf", dpi=300)
plt.show()


