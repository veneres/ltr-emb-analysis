import os
from collections import namedtuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np

import argparse

from pytorch_loader import IRDatasetLoader

# Avoid warning and possible deadlocks...
# "huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKEN_TYPE_PATH = "token_type_ids.csv"
INPUT_IDS_PATH = "input_ids.csv"
RUN_PATH = "run.csv"
EMBD_ZIPPED = "embd"

RunEntry = namedtuple("RunEntry", ["query_id", "doc_id", "score"])


def main():
    parser = argparse.ArgumentParser(description=
                                     f"""
                                     Create a folder for row analysis, it will contain:\n"
                                      - a folder {EMBD_ZIPPED} with all the hidden states of each pair query document 
                                      with name <query_id>_<doc_id>.npy
                                      - a csv {TOKEN_TYPE_PATH} containing the information about the token types
                                      - a csv {INPUT_IDS_PATH} containing the information about the input ids
                                     """
                                     )
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
    parser.add_argument("dir_out", type=str, help="Output dir")
    parser.add_argument("--batch_size", type=int, help="Batch size to use to speed up the computations", default=128)
    parser.add_argument("--device", type=str, help="Pytorch device", default="cuda")

    parser.add_argument("--compute_score",
                        help="Set to true if we want to compute the score, default true",
                        type=bool,
                        default=True)
    args = parser.parse_args()

    batch_size = args.batch_size

    output_dir = Path(args.dir_out).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_preproc_path = args.ir_dataset_preproc

    dataset_name = args.ir_dataset

    compute_score = args.compute_score

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, output_hidden_states=True)

    device = torch.device(args.device)

    model.to(device)

    dataset = IRDatasetLoader(dataset_name, tokenizer, irdataset_pt_file_name=dataset_preproc_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    token_type_ids_path = output_dir / TOKEN_TYPE_PATH
    input_ids_path = output_dir / INPUT_IDS_PATH
    run_path = output_dir / RUN_PATH
    embd_out = output_dir / EMBD_ZIPPED

    embd = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch_forward = {k: v.to(device) for k, v in batch.items() if k not in ["query_id", "doc_id", "labels"]}
            outputs = model(**batch_forward)

            query_ids = batch["query_id"]
            doc_ids = batch["doc_id"]
            if compute_score:
                scores = torch.softmax(outputs.logits, dim=1)[:, 1]

            real_batch_size = len(query_ids)
            n_hidden_states = len(outputs.hidden_states)

            col_run_entry = [name for name in RunEntry._fields]

            for i in range(real_batch_size):
                pair_embd = torch.stack([outputs.hidden_states[layer][i] for layer in range(n_hidden_states)])
                pair_embd = pair_embd.cpu().detach().numpy()
                embd[f"{query_ids[i]}_{doc_ids[i]}"] = pair_embd

                token_type_ids = batch["token_type_ids"][i].tolist()
                input_ids = batch["input_ids"][i].tolist()
                if compute_score:
                    score = scores[i].item()
                else:
                    score = np.nan
                query_id = query_ids[i].item()
                doc_id = doc_ids[i].item()

                run_entry = RunEntry(query_id=query_id, doc_id=doc_id, score=score)

                cols = [f"t_{dim}" for dim in range(len(token_type_ids))]
                # put it inside a list to make pandas understand it is a row
                df_token_type_ids = pd.DataFrame([token_type_ids], columns=cols)
                df_input_ids = pd.DataFrame([input_ids], columns=cols)
                df_run = pd.DataFrame([run_entry], columns=col_run_entry)

                # add query id and doc id information
                df_input_ids["query_id"] = df_token_type_ids["query_id"] = query_ids[i].item()
                df_input_ids["doc_id"] = df_token_type_ids["doc_id"] = doc_ids[i].item()

                write_header = not os.path.exists(token_type_ids_path)

                df_token_type_ids.to_csv(token_type_ids_path, mode='a', header=write_header, index=False)
                df_input_ids.to_csv(input_ids_path, mode='a', header=write_header, index=False)
                df_run.to_csv(run_path, mode='a', header=write_header, index=False)
    np.savez_compressed(embd_out, **embd)


if __name__ == '__main__':
    main()
