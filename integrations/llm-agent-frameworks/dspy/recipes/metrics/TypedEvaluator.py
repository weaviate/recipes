class Evaluator(dspy.Signature):
    """Evaluate the quality of a system's answer to a question according to a given criterion."""
    
    criterion: str = dspy.InputField(desc="The evaluation criterion.")
    question: str = dspy.InputField(desc="The question asked to the system.")
    ground_truth_answer: str = dspy.InputField(desc="An expert written Ground Truth Answer to the question.")
    predicted_answer: str = dspy.InputField(desc="The system's answer to the question.")
    rating: float = dspy.OutputField(desc="A float rating between 1 and 5")


def MetricWrapper(gold, pred, trace=None):
    alignment_criterion = "How aligned is the predicted_answer with the ground_truth?"
    return dspy.TypedPredictor(Evaluator)(criterion=alignment_criterion,
                                          question=gold.question,
                                          ground_truth_answer=gold.gold_answer,
                                          predicted_answer=pred.answer).rating
