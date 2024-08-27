## Embed and Search Text at Scale with Modal and Weaviate

In this demo, we built a full application that discovers analogies between Wikipedia articles by combining serverless infrastructure from Modal with the search and storage capabilities of Weaviate.

Resources:
* [Repository](https://github.com/modal-labs/vector-analogies-wikipedia)
* [Blog Post](https://weaviate.io/blog/modal-and-weaviate)

Taken from [`Hello, world! | Modal Docs`](https://modal.com/docs/examples/hello_world)

1. Install `modal` with `pip install modal`
2. Setup with `python3 -m modal setup`

This will set you up with a Modal API Key. You can then test it quickly with this simple app:

```python
import modal

app = modal.App("example-get-started")

@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))
```

There are two files in this repo:
1. `vllm_outlines_setup.py` (Run this with `modal deploy vllm_outlines_setup.py`)
2. `vllm_outlines_query.py` (Run this with `python3 vllm_outlines_query.py`)
