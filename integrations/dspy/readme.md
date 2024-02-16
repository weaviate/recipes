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

### LLM Metrics

```python
metricLM = dspy.OllamaLocal(model='mistral')

# Signature for LLM assessments.

class Assess(dspy.Signature):
    """Assess the quality of an answer to a question."""
    
    context = dspy.InputField(desc="The context for answering the question.")
    assessed_question = dspy.InputField(desc="The evaluation criterion.")
    assessed_answer = dspy.InputField(desc="The answer to the question.")
    assessment_answer = dspy.OutputField(desc="A rating between 1 and 5. Only output the rating and nothing else.")

def llm_metric(gold, pred, trace=None):
    predicted_answer = pred.answer
    question = gold.question
    
    print(f"Test Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")
    
    detail = "Is the assessed answer detailed?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    overall = f"Please rate how well this answer answers the question, `{question}` based on the context.\n `{predicted_answer}`"
    
    with dspy.context(lm=metricLM):
        context = dspy.Retrieve(k=5)(question).passages
        detail = dspy.ChainOfThought(Assess)(context="N/A", assessed_question=detail, assessed_answer=predicted_answer)
        faithful = dspy.ChainOfThought(Assess)(context=context, assessed_question=faithful, assessed_answer=predicted_answer)
        overall = dspy.ChainOfThought(Assess)(context=context, assessed_question=overall, assessed_answer=predicted_answer)
    
    print(f"Faithful: {faithful.assessment_answer}")
    print(f"Detail: {detail.assessment_answer}")
    print(f"Overall: {overall.assessment_answer}")
    
    
    total = float(detail.assessment_answer) + float(faithful.assessment_answer)*2 + float(overall.assessment_answer)
    
    return total / 5.0
```
