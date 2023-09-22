import argparse
import time

import ir_measures
import pandas as pd
import torch
import os

from collections import namedtuple

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from pytorch_loader import IRDatasetLoader

# Avoid warning and possible deadlocks...
# "huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

QueryDocRel = namedtuple("QueryDocRel", ["query_id", "doc_id", "score", "relevance"])


def main():
    parser = argparse.ArgumentParser(description=
                                     """
                                    Create dataframe with the relevance score for each query document pair from 
                                    a given dataset.
                                    Only the documents from dataset.scoreddocs_iter are taken into account, i.e. the 
                                    first 1000 docs retrieved from BM25 for each query.
                                    """)

    parser.add_argument("model", type=str, help="argument of AutoModelForSequenceClassification.from_pretrained()")
    parser.add_argument("tokenizer", type=str, help="argument of AutoTokenizer.from_pretrained()")
    parser.add_argument("ir_dataset", type=str, help="IR dataset to use from ir_dataset")
    parser.add_argument("ir_dataset_preproc",
                        type=str,
                        help="""
                        path to the IR dataset pre-processed to be used within the custom dataloader 
                        (see preproc_ir_datasets.py)
                        """
                        )

    parser.add_argument("out", type=str, help="Out file path containing the resulting dataframe")

    parser.add_argument("--batch_size", type=int, help="Batch size to use, default 1024", default=1024)
    parser.add_argument("--device", type=str, help="Device to use", default="cuda")

    args = parser.parse_args()

    batch_size = args.batch_size

    output = args.out

    dataset_name = args.ir_dataset
    dataset_preproc_path = args.ir_dataset_preproc

    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = IRDatasetLoader(dataset_name, tokenizer, irdataset_pt_file_name=dataset_preproc_path)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device(args.device)

    model.to(device)
    model.eval()

    # Collection of QueryDocRel used to create the final df
    entries_df = []

    # list to store all the predictions
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_forward = {k: v.to(device) for k, v in batch.items() if k not in ["query_id", "doc_id", "labels"]}
            outputs = model(**batch_forward)
            batch_queries_ids = batch["query_id"]
            batch_docs_ids = batch["doc_id"]
            rels = batch["labels"]
            scores = torch.softmax(outputs.logits, dim=1)[:, 1]

            for i, score in enumerate(scores):
                score = score.item()
                query_id = batch_queries_ids[i].item()
                doc_id = batch_docs_ids[i].item()
                rel = rels[i].item()

                entries_df.append(QueryDocRel(query_id=query_id, doc_id=doc_id, score=score, relevance=rel))

    pd.DataFrame(entries_df).to_csv(output, index=False)


if __name__ == '__main__':
    main()
