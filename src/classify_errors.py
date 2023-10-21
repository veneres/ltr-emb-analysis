import argparse
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import ir_datasets
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

DISTANCES = ["cosine", "l2", "js"]

EntryRes = namedtuple("EntryRes", ["measure", "valid_mean", "valid_sd", "test_ba", "acc"])


def prepare_dataset(df: pd.DataFrame, qrels: pd.DataFrame, norm=True):
    merged_df = df.set_index(["query_id", "doc_id"]).join(qrels.set_index(["query_id", "doc_id"]),
                                                          on=["query_id", "doc_id"], rsuffix="_type", how="inner")

    merged_df["relevance"] = merged_df["relevance"].astype(int)
    merged_df = merged_df.drop(columns="iteration")
    merged_df = merged_df.pivot(columns=["t"])
    merged_df.columns = [f"{c1}_{c2}" for c1, c2 in merged_df.columns]
    merged_df["score"] = merged_df["score_12"]
    merged_df["relevance"] = merged_df["relevance_12"]
    merged_df = merged_df.drop(columns=[f"score_{i}" for i in range(13)] + [f"relevance_{i}" for i in range(13)])
    merged_df = merged_df.reset_index()
    if norm:
        for query_id in merged_df["query_id"].unique():
            same_query_df = merged_df[merged_df["query_id"] == query_id]
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(same_query_df["score"].values.reshape(-1, 1))
            merged_df.loc[merged_df["query_id"] == query_id, "score"] = scaled_values.ravel()
    merged_df["bin_score"], bins = pd.cut(merged_df["score"], bins=[-1, 0.5, 10], labels=list(range(2)), retbins=True)
    merged_df["bin_relevance"] = pd.cut(merged_df["relevance"], bins=[-1, 2, 10], labels=list(range(2)))
    merged_df["bin_score"] = merged_df["bin_score"].astype(int)
    merged_df["bin_relevance"] = merged_df["bin_relevance"].astype(int)
    merged_df["y"] = (merged_df["bin_relevance"] - merged_df["bin_score"]).abs()

    X = merged_df.drop(columns=["y", "relevance", "score", "bin_relevance", "bin_score", "query_id", "doc_id"])
    # Drop NA columns when dataset is JS
    X = X.dropna(axis=1, how='all')
    X = X.replace([np.inf, -np.inf], 1)
    y = merged_df["y"]
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Probing classifier for mis-scoring")

    parser.add_argument("base_folder", type=str, help="Base folder containing the pairwise distances")
    parser.add_argument("out_folder", type=str, help="Base folder containing all the results")

    args = parser.parse_args()

    base_folder = Path(args.base_folder)
    out_folder = Path(args.out_folder)

    out_folder.mkdir(parents=True, exist_ok=True)

    msmarco_dl_2019 = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
    qrels_2019 = pd.DataFrame(msmarco_dl_2019.qrels_iter())

    msmarco_dl_2020 = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")
    qrels_2020 = pd.DataFrame(msmarco_dl_2020.qrels_iter())

    res = []
    for measure in DISTANCES:
        print("#" * 20)
        print(f"MEASURE: {measure}")
        df_train = pd.read_csv(f"{base_folder}/trec-dl-2019/{measure}.csv",
                               index_col=0,
                               dtype={"query_id": str, "doc_id": str})

        df_test = pd.read_csv(f"{base_folder}/trec-dl-2020/{measure}.csv",
                              index_col=0,
                              dtype={"query_id": str, "doc_id": str})

        train_X, train_y = prepare_dataset(df_train, qrels_2019)
        print(f"Number of miss-classified documents in train set: {np.sum(train_y == 1)}")
        print(f"Number of corrected-classified documents in train set: {np.sum(train_y == 0)}")
        test_X, test_y = prepare_dataset(df_test, qrels_2020)
        print(f"Number of miss-classified documents in test set: {np.sum(test_y == 1)}")
        print(f"Number of corrected-classified documents in test set: {np.sum(test_y == 0)}")

        parameters = {'max_leaf_nodes': [4, 8, 16, 32]}

        dtc = DecisionTreeClassifier(random_state=0)

        clf = GridSearchCV(dtc, parameters, scoring="balanced_accuracy")

        clf.fit(train_X, train_y)

        feat_imp = list(zip(clf.best_estimator_.feature_importances_, clf.best_estimator_.feature_names_in_))
        feat_imp.sort(key=lambda x: x[0], reverse=True)
        print("Feature Importance")
        print(feat_imp[:5])

        cv_mean_score = np.mean(clf.cv_results_['mean_test_score'])
        cv_sd_score = np.std(clf.cv_results_['mean_test_score'])

        print(clf.best_params_)

        ba_score = balanced_accuracy_score(test_y, clf.predict(test_X))
        acc = accuracy_score(test_y, clf.predict(test_X))

        fpr, tpr, thresholds = roc_curve(test_y, clf.predict_proba(test_X)[:, 1])
        roc_auc = auc(fpr, tpr)

        res_pickle = {"fpr": fpr,
                      "tpr": tpr,
                      "roc_auc": roc_auc}

        with open(out_folder / f"{measure}.pickle", "wb") as f:
            pickle.dump(res_pickle, f)

        res.append(EntryRes(measure=measure, valid_mean=cv_mean_score, valid_sd=cv_sd_score, test_ba=ba_score, acc=acc))

    pd.DataFrame(res).to_csv(out_folder / "acc.csv")


if __name__ == '__main__':
    main()
