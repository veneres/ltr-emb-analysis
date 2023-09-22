import argparse
import pandas as pd
import numpy as np
import torch
import scipy
import scipy.stats as st
from collections import namedtuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pytorch_loader import IRDatasetLoader

ResEntry = namedtuple("ResEntry",
                      ["query_id", "doc_id", "score", "min_sim", "max_sim", "avg_sim", "centroids_sim", "t"])

SPECIAL_TOKENS = {
    "bert-base-uncased": [101, 102, 0],
    "roberta-base": [0, 1, 2],
    "google/electra-base-discriminator": [101, 102, 0]
}


def fit_kernel(xx, yy, x, y):
    # fit a gaussian kernel using the scipy’s gaussian_kde method:
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return f


def compute_pca(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


def create_meshgrid(xmin, xmax, ymin, ymax):
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    return xx, yy


def compute_js_single_query_doc(data: np.ndarray,
                                idx_query_tokens: np.ndarray,
                                idx_doc_tokens: np.ndarray):
    pca_data = compute_pca(data)
    xx, yy = create_meshgrid(np.min(pca_data[:, 0]) - 1,
                             np.max(pca_data[:, 0]) + 1,
                             np.min(pca_data[:, 1]) - 1,
                             np.max(pca_data[:, 1]) + 1)

    x = pca_data[idx_doc_tokens, 0]
    y = pca_data[idx_doc_tokens, 1]
    f = fit_kernel(xx, yy, x, y)

    x2 = pca_data[idx_query_tokens, 0]
    y2 = pca_data[idx_query_tokens, 1]
    f2 = fit_kernel(xx, yy, x2, y2)

    js = scipy.spatial.distance.jensenshannon(f.ravel(), f2.ravel())

    return js, f, f2


def compute_dists(embd: np.ndarray,
                  query_tokens: np.ndarray,
                  query_selector: np.ndarray,
                  doc_tokens: np.ndarray,
                  doc_selector: np.ndarray,
                  query_id: str,
                  doc_id: str,
                  score: float,
                  t: int,
                  metric: str) -> ResEntry:
    if metric == "dot_prod":
        all_sim = query_tokens @ doc_tokens.T
    elif metric == "js":
        all_sim, f, f2 = compute_js_single_query_doc(embd, query_selector, doc_selector)
    else:
        all_sim = pairwise_distances(query_tokens, doc_tokens, metric=metric)

    min_sim = None
    max_sim = None
    avg_sim = None
    centroids_sim = all_sim
    if metric != "js":
        min_sim = np.min(all_sim)
        max_sim = np.max(all_sim)
        avg_sim = np.mean(all_sim)

        centroid_queries = np.mean(query_tokens, axis=0).reshape(1, -1)
        centroid_docs = np.mean(doc_tokens, axis=0).reshape(1, -1)
        if metric != "dot_prod":
            # returns vector of shape (1,1)
            centroids_sim = pairwise_distances(centroid_queries, centroid_docs, metric=metric)
        else:
            centroids_sim = centroid_queries @ centroid_docs.T
        centroids_sim = centroids_sim.item()

    return ResEntry(query_id=query_id,
                    doc_id=doc_id,
                    score=score,
                    min_sim=min_sim,
                    max_sim=max_sim,
                    avg_sim=avg_sim,
                    centroids_sim=centroids_sim,
                    t=t
                    )


def predict_and_compute_dist(batch, compute_score, device, metric, model, special_tokens):
    batch_forward = {k: v.to(device) for k, v in batch.items() if k not in ["query_id", "doc_id", "labels"]}
    outputs = model(**batch_forward)
    query_ids = batch["query_id"]
    doc_ids = batch["doc_id"]
    scores = None
    if compute_score:
        scores = torch.softmax(outputs.logits, dim=1)[:, 1]
    real_batch_size = len(query_ids)
    n_hidden_states = len(outputs.hidden_states)
    res = []
    for trans_block in range(n_hidden_states):
        for i in range(real_batch_size):
            embd = outputs.hidden_states[trans_block][i]
            embd = embd.cpu().detach().numpy()

            if "token_type_ids" not in batch:
                token_type_ids = []
                query_token = True
                for token_idx, token_id in enumerate(batch["input_ids"][i]):
                    token_type_ids.append(0 if query_token else 1)
                    if token_idx > 0 and token_id in special_tokens:
                        query_token = False
                token_type_ids = np.array(token_type_ids)
            else:
                token_type_ids = batch["token_type_ids"][i].cpu().detach().numpy()
            input_ids = batch["input_ids"][i].cpu().detach().numpy()

            score = np.nan
            if compute_score:
                score = scores[i].item()
            query_id = query_ids[i].item()
            doc_id = doc_ids[i].item()

            no_special_token_selector = ~np.isin(input_ids, special_tokens)

            query_selector = (token_type_ids == 0) & no_special_token_selector
            doc_selector = (token_type_ids == 1) & no_special_token_selector

            query_tokens = embd[query_selector, :]
            doc_tokens = embd[doc_selector, :]
            to_append = compute_dists(embd, query_tokens, query_selector, doc_tokens, doc_selector, query_id, doc_id,
                                      score, trans_block, metric)
            res.append(to_append)
    return res


def main():
    parser = argparse.ArgumentParser(description="Create dataframe for embeddings analysis")

    parser.add_argument("model", type=str, help="Argument of AutoModelForSequenceClassification.from_pretrained()")
    parser.add_argument("tokenizer", type=str, help="Argument of AutoTokenizer.from_pretrained()")
    parser.add_argument("ir_dataset", type=str, help="IR dataset to use from ir_dataset")
    parser.add_argument("ir_dataset_preproc",
                        type=str,
                        help="""
                                path to the IR dataset pre-processed to be used within the custom dataloader 
                                (see preproc_ir_datasets.py)
                                """
                        )

    parser.add_argument("out", type=str, help="Output file")

    parser.add_argument("--batch_size", type=int, help="Batch size to use to speed up the computations", default=128)
    parser.add_argument("--device", type=str, help="Pytorch device", default="cuda")

    parser.add_argument("--compute_score",
                        help="Set to true if we want to compute the score, default true",
                        type=bool,
                        default=True)

    parser.add_argument("--metric",
                        help="Metric to be used, possible choice: cosine, l2, dot_prod. Default cosine",
                        type=str,
                        default="cosine")
    args = parser.parse_args()

    metric = args.metric
    dataset_name = args.ir_dataset
    compute_score = args.compute_score
    dataset_preproc_path = args.ir_dataset_preproc
    batch_size = args.batch_size
    output = args.out

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    special_tokens = SPECIAL_TOKENS[args.tokenizer]
    model = AutoModelForSequenceClassification.from_pretrained(args.model, output_hidden_states=True)

    device = torch.device(args.device)

    model.to(device)

    model.eval()

    dataset = IRDatasetLoader(dataset_name, tokenizer, irdataset_pt_file_name=dataset_preproc_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    res = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            res.extend(predict_and_compute_dist(batch, compute_score, device, metric, model, special_tokens))

    pd.DataFrame(res).to_csv(output)


if __name__ == '__main__':
    main()
