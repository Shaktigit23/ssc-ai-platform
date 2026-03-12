from fastapi import FastAPI
from rag_query import ask_question
from mcq_generator import generate_mcq

app = FastAPI()

@app.get("/")
def home():
    return {"message":"SSC AI Learning Platform"}

@app.get("/ask")
def ask(query:str):
    return {"answer":ask_question(query)}

@app.get("/generate_mcq")
def mcq(text:str):
    return {"mcq":generate_mcq(text)}
