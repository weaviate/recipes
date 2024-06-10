import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM
import weaviate
import os

class AnswerQuestion(dspy.Signature):
    """Assess the context and answer the question. Use as much code as possible in your response."""

    question = dspy.InputField()
    chat_memory = dspy.InputField(desc="Will be empty if this is the first message the user has sent.")
    factual_context = dspy.InputField()
    answer = dspy.OutputField()

class UpdateMemory(dspy.Signature):
    """You are a memory manager for a coding assistant.
The coding assistant has a limited input window and thus must be selective about what information is saved in memory.
Given the previous memory and a new user message, update the memory.

IMPORTANT!!! IT IS EXTERMELY IMPORTANT THAT YOU PRESERVE CODE FOR MAINTAINING THE CONTEXT OF CODE EXCHANGED BETWEEN THE USER AND THE CODING ASSISTANT! THIS IS VERY IMPORTANT!!
    """

    previous_memory = dspy.InputField()
    user_message = dspy.InputField()
    updated_memory = dspy.OutputField(desc="IMPORTANT! DO NOT LOSE ANY CONTEXT OF THE PARTICULAR CODE DISCUSSED IN THE CONVERSATION!")

class FormulateSearchQuery(dspy.Signature):
    """Given the memory about a conversation and the most recent user message, formulate a search query to retrieve relevant factual information from a knowledge base."""

    memory = dspy.InputField()
    user_message = dspy.InputField()
    search_query = dspy.OutputField(desc="IMPORTANT! ONLY OUTPUT THE SEARCH QUERY AND NOTHING ELSE!") # Would be super cool to scale this to K queries

class MemGPT(dspy.Module):
    def __init__(self, retrieval_k=10):
        # This should definitely be moved outside.
        # Set desired LLM and your API Key as environment variables
        self.api_key = os.getenv("KEY")
        self.llm = os.getenv("LLM")
        if self.llm == "gpt-4-1106-preview":
            import openai
            openai.api_key = self.api_key
        print(self.llm)
        self.available_llms = {
            "command-r-plus": lambda: dspy.Cohere(model="command-r-plus", 
                                            api_key=self.api_key,
                                            max_input_tokens=32_000,
                                            max_tokens=4_000),
            "command-r": lambda: dspy.Cohere(model="command-r",
                                             api_key=self.api_key,
                                             max_input_tokens=32_000,
                                             max_tokens=4_000),
            "gemini-1.5-flash-latest": lambda: dspy.Google(model="gemini-1.5-flash-latest", 
                                                  api_key=self.api_key),
            "gpt-4-1106-preview": lambda: dspy.OpenAI(model="gpt-4-1106-preview", max_tokens=4_000)
        } # save state for meta API
        if self.llm in self.available_llms:
            self.base_llm = self.available_llms[self.llm]()
        else:
            raise ValueError(f"Model {self.llm} is not available in RAG with Recipes.")
            
        # Connect to Weaviate
        self.weaviate_client = weaviate.connect_to_local()
        self.memgpt_weaviate = self.weaviate_client.collections.get("MemGPTMemory")

        self.weaviate_recipes_index = WeaviateRM("WeaviateRecipesChunk",
                                            weaviate_client=self.weaviate_client)

        # Set DSPy defaults
        dspy.settings.configure(lm=self.base_llm, rm=self.weaviate_recipes_index)

        self.retrieve = dspy.Retrieve(k=retrieval_k)
        self.update_memory = dspy.ChainOfThought(UpdateMemory)
        self.formulate_query = dspy.ChainOfThought(FormulateSearchQuery)
        self.answer_question = dspy.ChainOfThought(AnswerQuestion)

    def get_meta(self):
        return self.available_llms

    def forward(self, message, sessionId):
        # RESPOND AND THEN UPDATE MEMORY

        # Get previous memory
        previous_memory = self.memgpt_weaviate.query.fetch_object_by_id(sessionId).properties["memory"]

        # Formulate search query based on previous memory and user message
        search_query = self.formulate_query(
            memory=previous_memory,
            user_message=message
        ).search_query
        
        # Retrieve relevant context based on the search query
        context = "".join(self.retrieve(search_query).passages)
        print(f"\nSearch Query\n{search_query}\n")

        # Generate response based on the question, previous memory, and retrieved context
        response = self.answer_question(
            question=message,
            chat_memory=previous_memory,
            factual_context=context
        ).answer

        print(f"\nResponse {response}\n")

        # Update memory based on the user message and previous memory
        updated_memory = self.update_memory(
            previous_memory=previous_memory,
            user_message=message
        ).updated_memory
        print(f"\nUpdated Memory\n{updated_memory}\n")

        # Update memory in Weaviate
        self.memgpt_weaviate.data.update(
            uuid=sessionId,
            properties={
                "memory": updated_memory
            }
        )
        print("\nUpdated Weaviate Memory\n")

        return dspy.Prediction(answer=response)