# Ingesting Data into Weaviate

This directory contains the code used to ingest the data into Weaviate.

## Amazon Product Dataset

You can download the Amazon product dataset from [here]([https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html#complete-data)).

Download the "CDs and Vinyl" dataset called "metadata" (not "reviews") and unzip it, then run code similar the following to split the data into multiple files:

```bash
split -l 10000 -d --additional-suffix=.json Amazon_Meta_CDs_Vinyl.json Amazon_Meta_CDs_Vinyl_
```

We have included a snippet of that with the first 10,000 lines of the CDs and Vinyl dataset in the Amazon_Meta_CDs_Vinyl_00.json

## Setting up to run the sample

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Also make sure that you start the weaviate docker images and ollama like below:

```bash
docker compose -f docker-compose.yaml up -d
ollama serve
ollama pull mxbai-embed-large:latest
ollama pull llama3.2:latest
```