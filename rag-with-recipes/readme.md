# RAG with Recipes

## How to run!

### Frontend
```bash
cd frontend
npm install
npm start
```

### Backend
```bash
cd weaviaterecipes
./set_env.sh --llm command-r --key YOUR-COHERE-KEY
```

This will set the api key, restore a Weaviate backup, and start a FastAPI server!
