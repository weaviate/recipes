from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/sendMessage")
async def send_message(msg: Message):
    return {"reply": "This is a response from the bot to your message: " + msg.message}
