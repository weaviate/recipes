# Welcome to the Weaviate Recipes repository!

This repo covers end-to-end examples on the various features and integrations with [Weaviate](www.weaviate.io)! Here is an outline of the folders:

## Similarity Search 🔎
[Similarity Search](https://github.com/weaviate/recipes/tree/main/similarity-search) shows how to run `nearText`, `nearObject` and `nearVector` queries in Weaviate. It is divided by the different providers:

* [Cohere](https://github.com/weaviate/recipes/blob/main/similarity-search/text2vec/similarity_search_cohere.ipynb)
* [Contextionary](https://github.com/weaviate/recipes/blob/main/similarity-search/text2vec/similarity_search_contextionary.ipynb)
* [HuggingFace](https://github.com/weaviate/recipes/blob/main/similarity-search/text2vec/similarity_search_huggingface.ipynb)
* [OpenAI](https://github.com/weaviate/recipes/blob/main/similarity-search/text2vec/similarity_search_openai.ipynb)
* [PaLM](https://github.com/weaviate/recipes/blob/main/similarity-search/text2vec/similarity_search_palm.ipynb)
* [Transformers](https://github.com/weaviate/recipes/blob/main/similarity-search/text2vec/similarity_search_transformers.ipynb)

## Hybrid Search ⚖️
[Hybrid Search](https://github.com/weaviate/recipes/tree/main/hybrid-search) allows you to combine keyword and vector search. The notebook covers how to run a hybrid search query, search on a specific property, add in a `where` filter, and how to search with an embedding. It is divided by the different providers:

* [Cohere](https://github.com/weaviate/recipes/blob/main/hybrid-search/hybrid_search_cohere.ipynb)
* [Contextionary](https://github.com/weaviate/recipes/blob/main/hybrid-search/hybrid_search_contextionary.ipynb)
* [HuggingFace](https://github.com/weaviate/recipes/blob/main/hybrid-search/hybrid_search_huggingface.ipynb)
* [OpenAI](https://github.com/weaviate/recipes/blob/main/hybrid-search/hybrid_search_openai.ipynb)
* [PaLM](https://github.com/weaviate/recipes/blob/main/hybrid-search/hybrid_search_palm.ipynb)
* [Transformers](https://github.com/weaviate/recipes/blob/main/hybrid-search/hybrid_search_transformers.ipynb)

## Generative Search ⌨️
[Generative Search](https://github.com/weaviate/recipes/tree/main/generative-search) allows you to improve your search results by piping them through LLM models. It is divided by the different providers:

* [Cohere](https://github.com/weaviate/recipes/blob/main/generative-search/generative_search_cohere.ipynb)
* [OpenAI](https://github.com/weaviate/recipes/blob/main/generative-search/generative_search_openai.ipynb)
* [PaLM](https://github.com/weaviate/recipes/blob/main/generative-search/generative_search_palm.ipynb)

## Integrations 🤝
[Integrations](https://github.com/weaviate/recipes/tree/main/integrations) with Weaviate

* LlamaIndex
  * [Episode 1: Data Loaders](https://github.com/weaviate/recipes/tree/main/integrations/llamaindex/data-loaders-episode1)
  * [Simple Query Engine](https://github.com/weaviate/recipes/tree/main/integrations/llamaindex/simple-query-engine)
  * [Sub Question Query Engine](https://github.com/weaviate/recipes/tree/main/integrations/llamaindex/sub-question-query-engine)


## Feedback ❓
Please note this is an ongoing project, and updates will be made frequently. If you have a feature you would like to see, please drop it in the [Weaviate Forum](https://forum.weaviate.io/c/general/4).
