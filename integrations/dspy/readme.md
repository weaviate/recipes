# DSPy and Weaviate!

Hey everyone, welcome to the DSPy integration cookbook with Weaviate!

Get started by importing data with `Weaviate-Import.ipynb`,

and then you are all setup to dive into `1.Getting-Started-with-RAG-in-DSPy`!

## Quick Nuggets

### RAG Program Example

```python
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("question, contexts -> answer")
    
    def forward(self, question):
        contexts = self.retrieve(question).passages
        prediction = self.generate_answer(question=question, contexts=contexts
        return dspy.Prediction(answer=prediction.answer)
```

### Ollama in DSPy

```python
import dspy
ollamaLM = dspy.OllamaLocal(model="mistral")
ollamaLM("Write a short poem")
```
