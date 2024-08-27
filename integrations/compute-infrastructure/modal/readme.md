## Embed and Search Text at Scale with Modal and Weaviate

In this demo, we built a full application that discovers analogies between Wikipedia articles by combining serverless infrastructure from Modal with the search and storage capabilities of Weaviate.

Resources:
* [Repository](https://github.com/modal-labs/vector-analogies-wikipedia)
* [Blog Post](https://weaviate.io/blog/modal-and-weaviate)

## Modal + vLLM + Outlines

There are additionally four files in this repo that demonstrate how to setup a vLLM server with Outlines Structured Decoding on Modal:

To achieve this run these commands:
```bash
modal run download_llama.py
modal deploy vllm_outlines_setup.py
```

Test with:
```bash
python3 vllm_outlines_query.py
```

Learn more about vLLM [here](https://github.com/vllm-project/vllm)!
Learn more about Outlines Structured Decoding [here](https://github.com/outlines-dev/outlines)!