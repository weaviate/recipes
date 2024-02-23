# Running Ollama with Weaviate!

Ollama is one of the easiest ways to run LLMs such as Mistral, Llama, or Gemma locally! Getting running with Ollama is as easy as:

1. Download Ollama, shown here: https://ollama.com/.
2. `ollama run llama2`

Now you can chat with LLMs hosted with Ollama in the terminal, or send POST requests as shown here:

```bash
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt":"Why is the sky blue?"
 }'
```

We are currently working on a `generative-ollama` module in Weaviate.

You can also access Ollama with their Python or JavaScript clients as shown here: https://ollama.com/blog/python-javascript-libraries.

You can access Ollama in Python through DSPy such as:

```
import dspy
ollamaLM = dspy.OllamaLocal(model="mistral")
ollamaLM("Write a short poem")
```
