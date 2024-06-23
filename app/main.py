from fastapi import FastAPI
from dotenv import load_dotenv
import os
load_dotenv()

required_env_vars = [
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY"
]

for var in required_env_vars:
    if not os.getenv(var):
        print(f"Environment variable {var} is not set.")
        exit(1)

app = FastAPI()

@app.get("/status")
async def read_root():
    return {"status": "OK"}