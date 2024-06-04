# RAG with Recipes backend

To control the LLM you are using set the model provider and api key in the CLI as follows:

```bash
python3 set_env_vars.py --llm command-r-plus --key AIfoobar
uvicorn main:app --reload
```
