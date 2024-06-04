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
    def __init__(self):
        # Set desired LLM and your API Key as environment variables
        llm = os.get_env("llm")
        api_key = os.get_env("key")
        self.available_llms = {
            "command-r-plus": dspy.Cohere(model="command-r-plus", api_key=api_key)
        } # save state for meta API
        self.base_llm = self.available_llms["Command-R+"]

        # Connect to Weaviate
        weaviate_recipes_index = WeaviateRM("WeaviateRecipesChunk",
                                            weaviate_client=weaviate.connect_to_local())

        # Set DSPy defaults
        dspy.settings.configure(lm=self.base_llm, rm=self.weaviate_recipes_index)

        self.retrieve = dspy.Retrieve()
        self.answer_question = dspy.ChainOfThought(AnswerQuestion)

    def get_meta(self):
        return self.available_llms

    def forward(self, chat_history):
        additional_context = self.retrieve() 
