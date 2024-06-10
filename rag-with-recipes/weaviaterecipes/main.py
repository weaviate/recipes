from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from language_programs import MemGPT
import weaviate
from weaviate.util import get_valid_uuid
from uuid import uuid4

program = MemGPT()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Message(BaseModel):
    message: str
    sessionId: str

@app.post("/create-session")
async def create_session():
    wevaiate_gfl_client = weaviate.connect_to_local()
    weaviate_blog_collection = wevaiate_gfl_client.collections.get("MemGPTMemory")
    random_uuid = get_valid_uuid(uuid4())
    uuid = weaviate_blog_collection.data.insert(
        uuid=random_uuid,
        properties={}
    )
    wevaiate_gfl_client.close()
    return {"session_id" : random_uuid}

@app.post("/sendMessage")
async def send_message(msg: Message):
    message = msg.message
    sessionId = msg.sessionId
    response = program(message=message,
                       sessionId=sessionId).answer
    print(response)
    return {"reply": response}
