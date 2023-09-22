import torch
from torch.utils.data import Dataset
import ir_datasets
from tqdm import tqdm


class IRDatasetLoader(Dataset):
    def __init__(self, ir_dataset_name: str, tokenizer, irdataset_pt_file_name: str):
        self.qrles = torch.load(irdataset_pt_file_name)  # qrels preprocessed by ir_datasets for re-ranking
        self.dataset = ir_datasets.load(ir_dataset_name)
        print("Creating doc store...")
        self.dataset_docstore = self.dataset.docs_store()

        print("Creating queries_store...")
        self.dataset_queriestore = {query.query_id: query.text for query in tqdm(self.dataset.queries_iter())}

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.qrles)

    def get_labels(self):
        return self.qrles[:, 2]

    def __getitem__(self, idx):
        qrel = self.qrles[idx, :]

        query_id = str(qrel[0].item())  # First element is query_id
        doc_id = str(qrel[1].item())  # Second element is doc_id
        relevance = qrel[2].item()  # Third element is relevance

        rel_docs_text = self.dataset_docstore.get(doc_id).text
        rep_query_text = self.dataset_queriestore[query_id]

        tokenized_text = self.tokenizer(rep_query_text,
                                        rel_docs_text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True)
        tokenized_text = {k: v[0] for k, v in tokenized_text.items()}
        tensor_qid = torch.tensor(int(query_id))
        tensor_did = torch.tensor(int(doc_id))
        return {**tokenized_text, "labels": relevance, "query_id": tensor_qid, "doc_id": tensor_did}

