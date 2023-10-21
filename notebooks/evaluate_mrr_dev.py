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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import ir_measures
import pandas as pd
from ir_measures import RR
from tqdm import tqdm
from collections import namedtuple
from pathlib import Path
from collections import defaultdict

# %%
import ir_datasets
DATA_PATH = "../../ltr-emb-analysis-data/results"

EntryRes = namedtuple("EntryRes", ["mrr", "model"])
models = ["monobert", "monoelectra", "monoroberta"]
main_path = Path()
res = defaultdict(list)
for model_name in tqdm(models):
    dir_path = Path(main_path / model_name)
    paths = [path for path in dir_path.iterdir()  if "scores_dev_small" in path.name]
    run_path = paths[0]
    try:
        general_df = pd.read_csv(run_path, index_col=0).astype(
            {"query_id": str, "doc_id": str, "relevance": int, "score": float})
    except:
        general_df = pd.read_csv(run_path).astype(
            {"query_id": str, "doc_id": str, "relevance": int, "score": float})
    run = general_df.drop(columns=['relevance'])
    metric_res = ir_measures.calc_aggregate([RR@10], ir_datasets.load("msmarco-passage/dev/small").qrels_iter(), run)
    res[model_name].append(EntryRes(mrr=metric_res[RR@10], model=model_name))

# %%
res

# %%
