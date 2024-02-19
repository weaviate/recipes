# DSPy and Weaviate!

Hey everyone, welcome to the DSPy integration cookbook with Weaviate!

This cookbook contains two parts:
1. Below is a list of `Quick Recipes`, a reference for useful DSPy functionality.
2. We also have notebooks numbered 1-K with full end-to-end DSPy tutorials. All the tutorials use the same data loaded into Weaviate. To get started from scratch, everything you need to start Weaviate is in the `docker-compose.yml` file (replace with your API key), the Weaviate blogs are chunked and loaded into this Weaviate instance in the `Weaviate-Import.ipynb` notebook. Once you complete these two steps, you are all set to dive into the end-to-end tutorials!

## Quick Recipes

Here are some quick recipes for DSPy.

### RAG Program Examples

#### Basic RAG

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

#### RAG with Reranking

```python
class Reranker(dspy.Signature):
    """Please rerank these documents."""
    
    context = dspy.InputField(desc="documents coarsely determined to be relevant to the question.")
    question = dspy.InputField()
    ranked_context = dspy.OutputField(desc="A ranking of documents by relevance to the question.")

class RAGwithReranker(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=5)
        self.reranker = dspy.ChainOfThought(Reranker)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        context = self.reranker(context=context, question=question).ranked_context
        pred = self.generate_answer(context=context, question=question).best_answer
        return dspy.Prediction(answer=pred)
```

#### RAG with Search Result Summarization

```python
class Summarizer(dspy.Signature):
    """Please summarize all relevant information in the context."""
    
    context = dspy.InputField(desc="documents determined to be relevant to the question.")
    question = dspy.InputField()
    summarized_context = dspy.OutputField(desc="A summarization of information in the documents that will help answer the quesetion.")

class RAGwithSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=5)
        self.summarizer = dspy.ChainOfThought(Summarizer)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        context = self.summarizer(context=context, question=question).summarized_context
        pred = self.generate_answer(context=context, question=question).best_answer
        return dspy.Prediction(answer=pred)
```

#### Multi-Hop RAG

```python
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

from dsp.utils import deduplicate

class MultiHopRAG(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_question = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_question[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.best_answer)
```

#### Multi-Hop RAG with Search Result Summarization

```python
class MultiHopRAGwithSummarization(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_question = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.summarizer = dspy.ChainOfThought(Summarizer)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_question[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            summarized_passages = self.summarizer(question=query, context=passages).summarized_context
            context.append(summarized_passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.best_answer)
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

### Tool Use Perspective with ReAct

```python
class RAGwithReAct(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ReAct(GenerateAnswer, tools=[self.retrieve])
    
    def forward(self, question):
        pred = self.generate_answer(question=question).best_answer
        return dspy.Prediction(answer=pred)
```
