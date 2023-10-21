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
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from pathlib import Path

# %%

DISTANCES = ["cosine", "l2", "js"]
BASE_FOLDER = Path("../../ltr-emb-analysis-data//classifiers/")

# %%
for model in ["monobert", "monoroberta", "monoelectra"]:
    df_res = pd.read_csv(BASE_FOLDER / f"{model}/acc.csv", index_col=0)
    print(df_res.to_latex())

# %%
plt.rcParams.update({
    "text.usetex": True,
    'font.size': 15,
    'figure.figsize':(9,6)
})
fig, ax = plt.subplots()
style_map = {
    "monobert": "-",
    "monoroberta": "--",
    "monoelectra": "dashdot",

}
palette = sns.color_palette("tab10", 3)
color_map = {
    "cosine": palette[0],
    "l2": palette[1],
    "js": palette[2],

}
model_name_dict = {
    "monobert": "MonoBERT",
    "monoroberta": "MonoROBERTa",
    "monoelectra": "MonoELECTRA",
}

measure_name_dict = {
    "cosine": "cosine",
    "l2": "l2",
    "js": "JSD",
}
for model in ["monobert", "monoroberta", "monoelectra"]:
    for measure in DISTANCES:
        with open(BASE_FOLDER / f"{model}/{measure}.pickle", "rb") as f:
            res = pickle.load(f)
            fpr = res["fpr"]
            tpr = res["tpr"]
            roc_auc = res["roc_auc"]
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{model_name_dict[model]}-\emph{{{measure_name_dict[measure]}}}")
            display.plot(ax=ax, ls=style_map[model],  color=color_map[measure])
fig.savefig("../plots/roc_curve.pdf", bbox_inches="tight")

# %%
