import base64

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, HTMLResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import ThingsRecognizer as tr

app = FastAPI()
app.mount("/static", StaticFiles(directory="Views"), name="static")

templates = Jinja2Templates(directory='Views')


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/recognizer")
async def read_item(type: str = Form(...), img: UploadFile = File(...)):
    try:
        contents = await img.read()
        with open(img.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return "Nie przesłano zdjęcia!"
    return tr.Recognize(type, img.filename)
