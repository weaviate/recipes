import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import os

class AnswerQuestion(dspy.Signature):
    """Assess the context and answer the question"""

    question = dspy.InputField()
    context = dspy.InputField()
    answer = dspy.OutputField()

class LanguageProgram(dspy.Module):
    def __init__(self, retrieval_k=10):
        # Set desired LLM and your API Key as environment variables
        self.llm = os.getenv("LLM")
        print(self.llm)
        self.api_key = os.getenv("KEY")
        self.available_llms = {
            "command-r-plus": lambda: dspy.Cohere(model="command-r-plus", 
                                                  api_key=self.api_key,
                                                  max_tokens=4_000)
        } # save state for meta API
        if self.llm in self.available_llms:
            self.base_llm = self.available_llms[self.llm]()
        else:
            raise ValueError(f"Model {self.llm} is not available in RAG with Recipes.")
            
        # Connect to Weaviate
        self.weaviate_recipes_index = WeaviateRM("WeaviateRecipesChunk",
                                            weaviate_client=weaviate.connect_to_local())

        # Set DSPy defaults
        dspy.settings.configure(lm=self.base_llm, rm=self.weaviate_recipes_index)

        self.retrieve = dspy.Retrieve(k=retrieval_k)
        self.answer_question = dspy.ChainOfThought(AnswerQuestion)

    def get_meta(self):
        return self.available_llms

    def forward(self, chat_history):
        context = self.retrieve(chat_history).passages
        pred = self.answer_question(context=context, question=chat_history).answer
        return dspy.Prediction(answer=pred)
