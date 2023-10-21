# Can Embeddings Analysis Explain Large Language Model Ranking?

Proof of concept of the work for CIKM 2023.

## Setup
To replicate the experiment you can use the Dockerfile provided that use a 
[Nvidia docker image](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html) 

To build the container you can use:

```bash
docker build --build-arg UID=$(id -u) --build-arg UNAME=$(id -un) -t ltr-emb-analysis:0.1 - < Dockerfile
```

Then, after having downloaded this repo inside the folder `<path-to-main-code-folder>`, we suggest to create
a folder for the data needed to replicate all the experiments at `<path-to-data-folder>`.

After this preamble you can run the container and open a shell to use with a command like this:

```bash
docker run --runtime nvidia  --user <my-user> --gpus <my-gpu-ids> --name <container-name> -i \
-v <path-to-main-code-folder>:/ltr-emb-analysis/ \
-v <path-to-data-folder>:/data/  \
-v <path-to-ir-dataset-home-folder>:/ir_datasets/ \
-v <path-to-huggingface-cache>:/hf/ \
-t ltr-emb-analysis:0.1 bash
```
where:

- `<my-user>` is your username
- `<my-gpu-ids>` [the restricting exposure of GPUs flag](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#setresgpuflag)
- `<container-name>` is a name of your choice for the container
- `<path-to-ir-dataset-home-folder>` is the path to the [home directory of ir dataset](https://github.com/allenai/ir_datasets#environment-variables)
- `<path-to-huggingface-cache>` is the path to the [home directory of huggingface cache](https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#cache-setup)


## Dataset preparation

For our convenience we preprocessed the ir_dataset in a big pytorch tensor. We know that this is not a feasible solution
for all the use cases but it was necessary for us to quickly load the triples during the training phases.


## Fine-tuning

The command to replicate the fine tuning are the following:

```bash

python finetune_mono_x.py bert msmarco-passage/train/judged /data/ir_dataset_torch/msmarco-passage_train_judged_triples.pt --batch_size=64 --save_after=10000 --device=cuda --output_dir=data/ltr-emb-analysis/models/monobert

python finetune_mono_x.py roberta msmarco-passage/train/judged /data/ir_dataset_torch/msmarco-passage_train_judged_triples.pt --batch_size=32 --save_after=10000 --device=cuda --output_dir=data/ltr-emb-analysis/models/monoroberta

python finetune_mono_x.py electra msmarco-passage/train/judged /data/ir_dataset_torch/msmarco-passage_train_judged_triples.pt --batch_size=32 --save_after=10000 --device=cuda --output_dir=data/ltr-emb-analysis/models/monoelectra

```

The pre-trained models are available at the following links:

- [monobert](https://huggingface.co/veneres/monobert)
- [monoroberta](https://huggingface.co/veneres/monoroberta)
- [monoelectra](https://huggingface.co/veneres/monoelectra)

If you want to check if the models performance you can run

```bash
python compute_scores.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/dev/small /data/ir_dataset_torch/msmarco-passage_dev_small.pt /data/ir_embedding_analysis/results/monobert/scores_dev_small.csv

python compute_scores.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/dev/small /data/ir_dataset_torch/msmarco-passage_dev_small.pt /data/ir_embedding_analysis/results/monoroberta/scores_dev_small.csv

python compute_scores.py /data/ltr-emb-analysis/models/monoelectra google/electra-base-discriminator msmarco-passage/dev/small /data/ir_dataset_torch/msmarco-passage_dev_small.pt /data/ir_embedding_analysis/results/monoelectra/scores_dev_small.csv
```


<!--
## RQ1

### Compute distances



python compute_dists.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/trec-dl-2019/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2019_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monobert/trec-dl-2019/cosine.csv --metric=cosine
python compute_dists.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/trec-dl-2019/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2019_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/bert/trec-dl-2019/l2.csv --metric=l2
python compute_dists.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/trec-dl-2019/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2019_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/bert/trec-dl-2019/dot_prod.csv --metric=dot_prod


python compute_dists.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/trec-dl-2020/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2020_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/bert/trec-dl-2020/cosine.csv --metric=cosine
python compute_dists.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/trec-dl-2020/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2020_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/bert/trec-dl-2020/l2.csv --metric=l2
python compute_dists.py /data/ltr-emb-analysis/models/monobert bert-base-uncased msmarco-passage/trec-dl-2020/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2020_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/bert/trec-dl-2020/dot_prod.csv --metric=dot_prod



python compute_dists.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/trec-dl-2019/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2019_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monoroberta/trec-dl-2019/cosine.csv --metric=cosine
python compute_dists.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/trec-dl-2019/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2019_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monoroberta/trec-dl-2019/l2.csv --metric=l2
python compute_dists.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/trec-dl-2019/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2019_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monoroberta/trec-dl-2019/dot_prod.csv --metric=dot_prod


python compute_dists.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/trec-dl-2020/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2020_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monoroberta/trec-dl-2020/cosine.csv --metric=cosine
python compute_dists.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/trec-dl-2020/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2020_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monoroberta/trec-dl-2020/l2.csv --metric=l2
python compute_dists.py /data/ltr-emb-analysis/models/monoroberta roberta-base msmarco-passage/trec-dl-2020/judged /data/ir_dataset_torch/msmarco-passage_trec-dl-2020_judged_qrel.pt /data/ltr-emb-analysis/pairwise_dist/monoroberta/trec-dl-2020/dot_prod.csv --metric=dot_prod

-->

If you reuse or like the work, cite:

```
@inproceedings{10.1145/3583780.3615225,
author = {Lucchese, Claudio and Minello, Giorgia and Nardini, Franco Maria and Orlando, Salvatore and Perego, Raffaele and Veneri, Alberto},
title = {Can Embeddings Analysis Explain Large Language Model Ranking?},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615225},
doi = {10.1145/3583780.3615225},
abstract = {Understanding the behavior of deep neural networks for Information Retrieval (IR) is crucial to improve trust in these effective models. Current popular approaches to diagnose the predictions made by deep neural networks are mainly based on: i) the adherence of the retrieval model to some axiomatic property of the IR system, ii) the generation of free-text explanations, or iii) feature importance attributions. In this work, we propose a novel approach that analyzes the changes of document and query embeddings in the latent space and that might explain the inner workings of IR large pre-trained language models. In particular, we focus on predicting query/document relevance, and we characterize the predictions by analyzing the topological arrangement of the embeddings in their latent space and their evolution while passing through the layers of the network. We show that there exists a link between the embedding adjustment and the predicted score, based on how tokens cluster in the embedding space. This novel approach, grounded in the query and document tokens interplay over the latent space, provides a new perspective on neural ranker explanation and a promising strategy for improving the efficiency of the models and Query Performance Prediction (QPP).},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {4150–4154},
numpages = {5},
keywords = {large language models, explainable artificial intelligence, embeddings analysis, text ranking},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}
```