# RAG with Recipes backend

To control the LLM you are using set the model provider and api key in the CLI as follows:

```bash
source set_env.sh --llm command-r-plus --key AIfoobar
uvicorn main:app --reload
```
