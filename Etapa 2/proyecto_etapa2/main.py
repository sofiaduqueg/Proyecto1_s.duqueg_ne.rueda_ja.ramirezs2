from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import subprocess

app = FastAPI()

class User(BaseModel):
   review:str

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })

# @app.post('/analisis_sentimiento')
# async def analisis_sentimiento(review: str = Form(...)):
#     return templates.TemplateResponse("index.html", {
#         "review": review
#     })

# @app.post("/analisis_sentimiento", response_class=HTMLResponse)
# async def analisis_sentimiento(request: Request, review: str = Form(...)):
#     result = review
#     return result

@app.post("/analisis_sentimiento/", response_model=User)
async def analisis_sentimiento(rw: str = Form(...)):
    with open('./static/assets/txt/review.txt', 'w') as f:
        f.write(rw)
    os.system('python script.py')
    with open('./static/assets/txt/sentiment.txt') as f:
        lines = f.readlines()
    return {"review": str(lines[0])}

