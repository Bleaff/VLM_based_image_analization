# app/api.py
import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from .config import load_config, load_prompts
from .ollama_client import OllamaClient
from .pipeline import Pipeline

# load config & prompts at startup
CFG = load_config(os.environ.get("APP_CONFIG_PATH", "./config/config.yml"))
PROMPTS = load_prompts(CFG.prompts_file)
OLLAMA = OllamaClient(CFG.ollama.url, CFG.ollama.timeout_seconds)
PIPE = Pipeline(CFG, PROMPTS, OLLAMA)

app = FastAPI(title="Modular Qwen Pipeline")

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(None),
    action: str = Form(...),
    answers: str = Form(None),
    dialog_state: str = Form(None),
    stream: str = Form("false")
):
    if action not in CFG.service.allowed_actions:
        raise HTTPException(status_code=400, detail="unsupported action")

    stream_mode = str(stream).lower() == "true"

    # parse state/answers
    prev_state = {}
    if dialog_state:
        try:
            prev_state = json.loads(dialog_state)
        except Exception:
            raise HTTPException(status_code=400, detail="dialog_state must be valid JSON")

    if action == "new":
        if image is None:
            raise HTTPException(status_code=400, detail="image required for action=new")
        img_bytes = await image.read()
        res = await PIPE.ask_questions(img_bytes, image.filename, stream=stream_mode)
        # include dialog_state minimal
        response_state = {"stage":"asking", "image_description": res.get("image_description")}
        return JSONResponse({**res, "dialog_state": response_state})

    else:  # action == 'answer'
        if not answers:
            raise HTTPException(status_code=400, detail="answers required for action=answer")
        try:
            answers_obj = json.loads(answers)
        except Exception:
            raise HTTPException(status_code=400, detail="answers must be valid JSON")
        image_description = prev_state.get("image_description", "No image description available.")
        res = await PIPE.finalize(image_description, answers_obj, stream=stream_mode)
        # add state
        response_state = {"stage":"final"}
        return JSONResponse({**res, "dialog_state": response_state})
