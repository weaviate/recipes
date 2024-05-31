from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dspy
import cohere
import weaviate
from weaviate.util import get_valid_uuid
from uuid import uuid4

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Weaviate and configure LLM

cohere_key = ""

command_nightly = dspy.Cohere(model="command-nightly",max_tokens=1000, api_key=cohere_key)

dspy.settings.configure(lm=command_nightly)

weaviate_client = weaviate.connect_to_local()

chat_snippets = weaviate_client.collections.get("ChatMessage")

class AnswerWithPersona(dspy.Signature):
    """Respond to the most recent chat message with a response consistent with the given persona."""

    persona = dspy.InputField()
    chat_history = dspy.InputField()
    additional_context = dspy.InputField(desc="Retrieval to help with the most recent question.")
    response = dspy.OutputField(prefix = "response:")

class RAGwithPersona(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rag_with_persona = dspy.Predict(AnswerWithPersona)
    
    def forward(self, persona, chat_history):
        response = self.rag_with_persona(persona=persona, chat_history=chat_history).response
        return dspy.Prediction(response=response)

program = RAGwithPersona()

class ChatbotData(BaseModel):
    messages: list
    topText: str

@app.post("/RAGwithPersona")
async def chatbot_endpoint(data: ChatbotData):
    # Access the chat history and top text from the request data
    chat_history = data.messages
    persona = data.topText

    # Parse the chat history into a string
    parsed_history = ", ".join([f"{chat['sender']}: {chat['text']}" for chat in chat_history])
    program_output = program(persona=persona, chat_history=parsed_history)
    response = program_output.response
    user_chat_id = get_valid_uuid(uuid4())
    chat_snippets.data.insert(
        properties={
            "author": "user",
            "content": parsed_history[-1]
        },
        uuid=user_chat_id
    )
    program_chat_id = get_valid_uuid(uuid4())
    chat_snippets.data.insert(
        properties={
            "author": "RAGwithPersona",
            "content": response
        },
        uuid=program_chat_id
    )
    response_data = {
        "response": response
    }
    return response_data
