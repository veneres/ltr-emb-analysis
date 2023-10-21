# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import scipy.stats
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.rcParams.update({
    "text.usetex": True,
    'font.size': 10,
    'figure.figsize':(15,6)
})

# %%
from pathlib import Path

DATA_PATH = "../../ltr-emb-analysis-data/pairwise_dist/monobert/trec-dl-2019"
res_dir = Path(DATA_PATH)
measure = "cosine"

# %%
# Share both X and Y axes with all subplots
fig, axs = plt.subplots(3, 3, sharex='all')
label_coords = (-0.13,0.5)

title_dict = {"min_sim": "min", "avg_sim":"avg", "max_sim":"max"}

for row_id, measure in enumerate(["cosine", "l2"]):
    for col_id, measure_type in enumerate(["min_sim", "avg_sim", "max_sim"]):
        df_2019 = pd.read_csv(res_dir / f"{measure}.csv", index_col=0, dtype={"query_id": str, "doc_id": str})
        binned_score = ["High score" if high_score else "Low score" for high_score in df_2019["score"] > 0.5]
        ax = axs[row_id, col_id]
        sns.lineplot(df_2019, x="t", y=measure_type, hue=binned_score, errorbar=("pi", 50), legend=False, ax=ax)
        ax.set(xlabel='T', ylabel="", title=f'\emph{{{title_dict[measure_type]}-{measure}}}')
        ax.set_xticks(range(0, 13, 4), range(0, 13, 4))
        ax.get_yaxis().set_label_coords(*label_coords)

for col_id, measure in enumerate(["cosine", "l2"]):
    ax = axs[2, col_id]
    df_2019 = pd.read_csv(res_dir / f"{measure}.csv", index_col=0, dtype={"query_id": str, "doc_id": str})
    binned_score = ["High score" if high_score else "Low score" for high_score in df_2019["score"] > 0.5]
    sns.lineplot(df_2019, x="t", y="centroids_sim", hue=binned_score, errorbar=("pi", 50), legend=False, ax=ax)
    ax.set(xlabel='T', ylabel="", title=f'\emph{{center-{measure}}}')
    ax.set_xticks(range(0, 13, 4), range(0, 13, 4))
    ax.get_yaxis().set_label_coords(*label_coords)

ax = axs[2, 2]
df_2019 = pd.read_csv(res_dir / f"js.csv", index_col=0, dtype={"query_id": str, "doc_id": str})
binned_score = ["High score" if high_score else "Low score" for high_score in df_2019["score"] > 0.5]
sns.lineplot(df_2019, x="t", y="centroids_sim", hue=binned_score, errorbar=("pi", 50), legend=False, ax=ax)
ax.set(xlabel='T', ylabel="", title=r'\emph{JSD}')
ax.set_xticks(range(0, 13, 4), range(0, 13, 4))
ax.get_yaxis().set_label_coords(*label_coords)
fig.savefig("../plots/distances.pdf", bbox_inches='tight')

# %%
ax = sns.histplot(data=df_2019[df_2019["t"] == 0], x="score")
ax.set(xlabel="Predicted score", ylabel="Count")
plt.savefig(f"../plots/score_dist.pdf",bbox_inches='tight')


# %%
def statistic(x, y):
    return np.mean(x) - np.mean(y)

for measure in ["cosine", "l2", "js"]:
    for value in ["min_sim","avg_sim", "max_sim", "centroids_sim"]:
        if measure=="js" and value != "centroids_sim":
            continue
        all_different = True
        for t in range(12):
            df = pd.read_csv(res_dir / f"{measure}.csv", index_col=0, dtype={"query_id": str, "doc_id": str})
            # get a specific transformer block
            df = df[df["t"] == t]
            # get all the pair with a high score
            high_score = df[df["score"] > 0.5][value].values
            low_score = df[df["score"] <= 0.5][value].values
            res = scipy.stats.permutation_test([high_score, low_score], statistic, random_state=42, vectorized=False, permutation_type="independent", alternative="less")
            if res.pvalue > 0.01:
                all_different = False
                break
        if all_different:
            print(f"{measure}-{value}: difference is significant for all the transformer blocks")


 # %%
 for t in range(12):
    df = pd.read_csv(res_dir / f"js.csv", index_col=0, dtype={"query_id": str, "doc_id": str})
    # get a specific transformer block
    df = df[df["t"] == t]
    # get all the pair with a high score
    high_score = df[df["score"] > 0.5]["centroids_sim"].values
    low_score = df[df["score"] <= 0.5]["centroids_sim"].values
    high_score[high_score == np.inf] = 1
    low_score[low_score == np.inf] = 1
    res = scipy.stats.permutation_test([high_score, low_score], statistic, random_state=42, vectorized=False, permutation_type="independent", alternative="less")
    if res.pvalue < 0.01:
        print(f"{measure} - {value} - {t}")

# %%
print(pd.read_csv(res_dir / f"js.csv", index_col=0, dtype={"query_id": str, "doc_id": str}))

# %%
