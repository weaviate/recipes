import modal

volume = modal.Volume.from_name("cagra-cu-vs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "cuvs-cu12",
            "cupy==13.2.0",
            "sentence-transformers==3.0.1",
            "transformers==4.44.2",
            "torch==2.1.1",
        ],
        extra_index_url="https://pypi.nvidia.com"
    )
)

MINUTES = 60
HOURS = 5 * MINUTES

app = modal.App(image=image) # add secrets here when using an embeddings API (ToDo)

class CAGRA:
    @modal.enter()
    def init_cagra_graph(self):
        # Init CAGRA
        from cuvs.neighbors import cagra
        self.params = cagra.IndexParams(intermediate_graph_degree=128,
                                   graph_degree=64)
        # Init Embedding Service
        # ToDo -- move this to download_models.py, I think these are really small models (~500 MB) so not needed yet
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        
    @modal.method()
    def build_cagra_from_embeddings(self, embeddings):
        # ToDo find API to add to the existing graph
      
        # wrap the input vector in a CUDA Numpy array
        import cupy as cp
        cp_embeddings = cp.asarray(embeddings)

        self.cagra_index = cagra.build_index(cp_embeddings, self.params)

    # Should this be a generator? `is_generator=True`
    @modal.method()
    def search_cuvs_cagra(self, query, top_k = 5):
        # ToDo, extend to test batch queries
      
        import time
        import torch

        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.model.encode(query, convert_to_tensor=True)

        start_time = time.time()
        hits = self.cagra.search(self.params, self.cagra_index, question_embedding[None], top_k)
        end_time = time.time()

        score_tensor = torch.as_tensor(hits[0], device='cuda')
        index_tensor = torch.as_tensor(hits[1], device='cuda')
        return score_tensor, index_tensor

# Thoughts on Future Work
'''

# Still need to understand the best way to move memory around from Weaviate <> CAGRA on Modal

Could be something like this similar to how Llama3 inference works:

```
Maytbe this approach when only loading in the CAGRA graph for Inference

try:
    volume = modal.Volume.lookup("cagra-cu-vs", create_if_missing=True)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal to run download_llama.py")

# Save CAGRA graph to Modal volume
@app.function(volumes={"/data": volume}, timeout=4 * HOURS)
def save_cagra_graph():
    pass
    
Load CAGRA graph from Weaviate
@app.function(volumes={"/data": volume}, timeout=4 * HOURS)
def load_cagra_graph(weaviate_client, weaviate_collection_name):
    pass


@app.local_entrypoint()
def main(
    weaviate_client,
    weaviate_collection_name
):
    load_cagra_graph.remote(weaviate_client, weaviate_collection_name)
'''
