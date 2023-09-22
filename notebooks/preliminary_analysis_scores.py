# ---
# jupyter:
#   jupytext:
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import os
os.getcwd()

# %%
monobert_dev_scores = "../results/monobert/msmarco-passage_dev_small.csv"

# %%
plt.rcParams.update({
    "text.usetex": True
})

# %%
msmarco_dev_scores = pd.read_csv(monobert_dev_scores, index_col=0,
                                 dtype={'query_id': str, 'doc_id': str})
ax = sns.histplot(data=msmarco_dev_scores, x="score", bins=20, stat="percent", log_scale=(False, True))
ax.set_xlabel("MonoBERT Score")
ax.set_ylabel("Percentage (log scale)")
plt.grid(visible=True)
plt.savefig("../plots/cumulative_bert_score.png")

# %%
