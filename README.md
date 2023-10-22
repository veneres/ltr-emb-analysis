# Can Embeddings Analysis Explain Large Language Model Ranking?

Proof of concept of the work for CIKM 2023.


## Preliminary

All the suggested paths are structured in a way that helps the replication with the provided notebooks. This is because
we normally loop in the same folder to replicate the analysis for the different datasets and different models considered.

The code of the notebooks to create the tables and figures are inside the folder `notebooks` and the can be converted 
in actual jupyter notebook using [jupytext](https://github.com/mwouts/jupytext).

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

To preprocess a dataset you can use a command like this:

```bash
python preproc_ir_datasets.py "msmarco-passage/train/judged" "<path-to-data-folder>/ir-datasets-pt/msmarco-passage_train.pt"
```

Where `"msmarco-passage/train/judged"` is the `ir_dataset` tagname of the dataset used for training and 
`<path-to-data-folder>/ir-datasets-pt/` is the path where to store the preprocessed `ir-datasets`.

## Fine-tuning

The command to replicate the fine tuning are the following:

```bash

python finetune_mono_x.py <llm> msmarco-passage/train/judged "<path-to-data-folder>/ir-datasets-pt/msmarco-passage_train.pt" --batch_size=64 --save_after=10000 --device=cuda --output_dir="<path-to-data-folder>/models/<llm_name>"
```

Where `<llm>` is the tagname for the LLM used. Valid tagnames for the script are:

- `bert`
- `roberta`
- `electra`


Finally, `"<path-to-data-folder>/models/<llm_name>"` is the path where to store the final model.

The pre-trained models are available at the following links:

- [monobert](https://huggingface.co/veneres/monobert)
- [monoroberta](https://huggingface.co/veneres/monoroberta)
- [monoelectra](https://huggingface.co/veneres/monoelectra)

If you want to check if the models performance you can run

```bash
python compute_scores.py "<path-to-data-folder>/models/<llm_name>" <tokenizer> msmarco-passage/dev/small "<path-to-data-folder>/ir-datasets-pt/msmarco-passage_dev_small.pt"  <path-to-data-folder>/results/<llm_name>.csv 
```

Where `<tokenizer>` is the tokenizer huggingface name for the model used, `msmarco-passage/dev/small` is the dataset 
used for the validate the quality of the trained model and 
`<path-to-data-folder>/ir-datasets-pt/msmarco-passage_dev_small.pt` is the preprocessed version of the same dataset.
and `<path-to-data-folder>/results/<llm_name>.csv` is the output path where to save the CSV with the results. 


## RQ1 - RQ2

## Compute distances

To compute the distances between the embeddings you can use the `compute_dists.py` script in the following way:

```bash
python compute_dists.py "<path-to-data-folder>/models/<llm_name>" `<tokenizer>` <trec-dl-name> "<path-to-data-folder>/ir-datasets-pt/<trec-dl-name>" "<path-to-data-folder>/pairwise_dist/<model-name>/<trec-dl-name>/<metric-name>.csv" --metric=<metric-name>
```

Where the `<trec-dl-name>` is the ir_dataset name for the two datasets under consideration, namely 
`msmarco-passage/trec-dl-2019/judged` and `msmarco-passage/trec-dl-2020/judged`.
`"<path-to-data-folder>/pairwise_dist/<model-name>/<trec-dl-name>/<metric-name>.csv"` is instead the default path where
to output the distances results. 
Finally `<metric-name>` is one of the possibile metric used. The available metric are: 
- `cosine`
- `l2`
- `js`

## Classify errors

The script to classify the errors made by the models is `classify_errors.py` and has to be used in the following way:

```bash
python classify_errors.py "<path-to-data-folder>/pairwise_dist/<model-name>/<trec-dl-name>/<metric-name>.csv" "<path-to-data-folder>/classifiers/<model-name>/acc.csv"
```
where `"<path-to-data-folder>/classifiers/<model-name>/acc.csv"` is the path where to store the results of the evaluation.

*Note*: in the published paper there is a subtle typo in the following phrase:
"We then evaluate the classification accuracy on datasets created from trec-dl-20*19*".
Of course, the test has been dataset w.r.t the training one, we perform the evaluation on "trec-dl-20*20*"

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