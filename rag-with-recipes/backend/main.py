from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from language_program import LanguageProgram

program = LanguageProgram()

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

@app.post("/sendMessage")
async def send_message(msg: Message):
    message = msg.message
    return {"reply": "This is a response from the bot to your message: " + message[0]}
