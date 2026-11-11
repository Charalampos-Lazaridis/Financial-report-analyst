from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from app.analyst import MAX_QUESTION_LENGTH, AskResponse, ask

app = FastAPI(title="Financial Report Analyst")

class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    user: str | None = Field(default=None, max_length=40)  # a name turns on memory; none = no memory

@app.get("/")
def health() -> dict:
    return {"status": "service is online"}

@app.post("/ask")
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.question, request.user)