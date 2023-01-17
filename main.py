from fastapi import FastAPI, Form
from starlette.responses import FileResponse

import PersonRecognizer as pr

app = FastAPI()


@app.get("/")
async def root():
    return FileResponse('Views/index.html')


@app.get("/recognizer")
async def StartCounting(type: str):
    return pr.Recognize(type)
