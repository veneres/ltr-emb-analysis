import argparse
from collections import defaultdict

from tqdm import tqdm
import ir_datasets
import torch


def create_docpairs(dataset, max_q):
    res = []
    qids_set = set()
    for docpair in tqdm(dataset.docpairs_iter(), total=dataset.docpairs_count()):
        query_id = int(docpair.query_id)

        if query_id not in qids_set and max_q is not None and len(qids_set) >= max_q:  # queries_cutoff
            continue
        entry = [query_id, int(docpair.doc_id_a), 1]
        res.append(entry)
        qids_set.add(query_id)

        entry = [query_id, int(docpair.doc_id_b), 0]
        res.append(entry)
        qids_set.add(query_id)
    return res


def create_scoreddocs(dataset: ir_datasets.Dataset, max_q: int):
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)

    res = []
    qids_set = set()
    for scoreddoc in tqdm(dataset.scoreddocs_iter(), total=dataset.scoreddocs_count()):
        query_id = int(scoreddoc.query_id)

        if query_id not in qids_set and max_q is not None and len(qids_set) >= max_q:  # queries_cutoff
            continue

        relevance = qrels[scoreddoc.query_id][scoreddoc.doc_id] if scoreddoc.doc_id in qrels[scoreddoc.query_id] else 0
        doc_id = int(scoreddoc.doc_id)
        entry = [query_id, doc_id, relevance]
        res.append(entry)
        qids_set.add(query_id)
    return res


def create_qrels(dataset: ir_datasets.Dataset, max_q: int):
    res = []
    qids_set = set()
    for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count()):
        query_id = int(qrel.query_id)
        if query_id not in qids_set and max_q is not None and len(qids_set) >= max_q:  # queries_cutoff
            continue
        relevance = qrel.relevance
        doc_id = int(qrel.doc_id)
        entry = [query_id, doc_id, relevance]
        res.append(entry)
        qids_set.add(query_id)
    return res


def main():
    parser = argparse.ArgumentParser(description="""
    Converting a dataset from ir_dataset to be pre-processed for re-ranking.
    The transformation transform a ir_dataset to a tensor where each row represents a scoreddoc and have only 3 columns:
    - query id
    - doc id
    - relevance (taken from qrels or scorreddocs or docpairs)
    """)
    parser.add_argument("ir_dataset", type=str, help="ir_dataset to be transformed")
    parser.add_argument("output_name", type=str, help="File name of the transformed tensor")
    parser.add_argument("--max_q", type=int, help="Slice the dataset at max_q queries", default=None)
    parser.add_argument("--generator", type=str,
                        help="Chose to use the docpairs or scorreddocs or qrels during the loop generator, default"
                             "scorreddocs", default="scorreddocs")
    args = parser.parse_args()

    dataset = ir_datasets.load(args.ir_dataset)

    max_q = args.max_q

    if args.generator == "docpairs":
        res = create_docpairs(dataset, max_q)
    elif args.generator == "scorreddocs":
        res = create_scoreddocs(dataset, max_q)
    elif args.generator == "qrels":
        res = create_qrels(dataset, max_q)
    else:
        raise Exception(f"Illegal argument: {args.generator=}")

    res = torch.tensor(res)

    print(f"Number of entries: {len(res)}")

    torch.save(res, args.output_name)


if __name__ == '__main__':
    main()
