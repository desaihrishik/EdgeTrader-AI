from fastapi import APIRouter
from pydantic import BaseModel
from src.llm_client import ask_llm

router = APIRouter(prefix="/api/llm")

class ObserveRequest(BaseModel):
    setup: str

@router.post("/observe")
def observe_llm(data: ObserveRequest):
    analysis = ask_llm(
        f"You are a trading assistant. Analyze this setup:\n{data.setup}"
    )
    return {"analysis": analysis}


class ChoiceRequest(BaseModel):
    choice: str
    context: str

@router.post("/evaluate_choice")
def evaluate_choice(data: ChoiceRequest):
    result = ask_llm(
        f"User choice: {data.choice}\nContext: {data.context}\nEvaluate this decision."
    )
    return {"evaluation": result}


class ExplainMoreRequest(BaseModel):
    question: str

@router.post("/explain_more")
def explain_more(data: ExplainMoreRequest):
    result = ask_llm(
        f"Explain more about: {data.question}"
    )
    return {"response": result}
