# DSPy Recipes

### LLM Metric

```python
class Evaluator(dspy.Signature):
    """Evaluate the quality of a system's answer to a question according to a given criterion."""
    
    context = dspy.InputField(desc="The context for answering the question.")
    criterion = dspy.InputField(desc="The evaluation criterion.")
    question = dspy.InputField(desc="The question asked to the system.")
    ground_truth_answer = dspy.InputField(desc="An expert written ground truth answer to the question.")
    predicted_answer = dspy.InputField(desc="The system's answer to the question.")
    rating = dspy.OutputField(desc="A rating between 1 and 5. IMPORTANT!! Only output the rating as an `int` and nothing else.")

class RatingParser(dspy.Signature):
    """Parse the rating from a string."""
    
    raw_rating_response = dspy.InputField(desc="The string that contains the rating in it.")
    rating = dspy.OutputField(desc="An integer valued rating.")
    
class Summarizer(dspy.Signature):
    """Summarize the information provided in the search results in 5 sentences."""
    
    question = dspy.InputField(desc="a question to a search engine")
    context = dspy.InputField(desc="context filtered as relevant to the query by a search engine")
    summary = dspy.OutputField(desc="a 5 sentence summary of information in the context that would help answer the question.")

class MetricProgram(dspy.Module):
    def __init__(self):
        self.evaluator = dspy.ChainOfThought(Evaluator)
        self.rating_parser = dspy.Predict(RatingParser)
        self.summarizer = dspy.ChainOfThought(Summarizer)
    
    def forward(self, gold, pred, trace=None):
        # Todo add trace to interface with teleprompters
        predicted_answer = pred.answer
        question = gold.question
        ground_truth_answer = gold.answer
        
        detail = "Is the assessed answer detailed?"
        faithful = "Is the assessed answer factually supported by the context?"
        ground_truth = f"The ground truth answer to the question: {question} is given as {ground_truth_answer}. How aligned is this answer? {predicted_answer}"
        
        # Judgement
        with dspy.context(lm=command_nightly):
            context = dspy.Retrieve(k=10)(question).passages
            # Context Summary
            context = self.summarizer(question=question, context=context).summary
            raw_detail_response = self.evaluator(context=context, 
                                 criterion=detail,
                                 question=question,
                                 ground_truth_answer=ground_truth_answer,
                                 predicted_answer=predicted_answer).rating
            raw_faithful_response = self.evaluator(context=context, 
                                 criterion=faithful,
                                 question=question,
                                 ground_truth_answer=ground_truth_answer,
                                 predicted_answer=predicted_answer).rating
            raw_ground_truth_response = self.evaluator(context=context, 
                                 criterion=ground_truth,
                                 question=question,
                                 ground_truth_answer=ground_truth_answer,
                                 predicted_answer=predicted_answer).rating
        
        # Structured Output Parsing
        with dspy.context(lm=gpt4):
            detail_rating = self.rating_parser(raw_rating_response=raw_detail_response).rating
            faithful_rating = self.rating_parser(raw_rating_response=raw_faithful_response).rating
            ground_truth_rating = self.rating_parser(raw_rating_response=raw_ground_truth_response).rating
        
        total = float(detail_rating) + float(faithful_rating)*2 + float(ground_truth_rating)
    
        return total / 5.0

toy_ground_truth_answer = """
Cross encoders score the relevance of a document to a query. They are commonly used to rerank documents.
"""

test_example = dspy.Example(question="What do cross encoders do?", answer=toy_ground_truth_answer)
test_pred = dspy.Example(answer="They re-rank documents.")

llm_metric = MetricProgram()
llm_metric_rating = llm_metric(test_example, test_pred)
```
