# pipeline_app.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
import os
import httpx
import json
import asyncio

app = FastAPI(title="Pipeline → Ollama (stream-aware)")

OLLAMA_URL = os.environ.get("APP_OLLAMA_URL", "http://ollama:11434/api/generate")

async def stream_ollama_and_collect(client: httpx.AsyncClient, payload, files=None, timeout=120.0):
    """
    Отправляет запрос в ollama и читает стрим-по-строке JSON.
    Возвращает (final_text, list_of_chunks, last_obj)
    """
    chunks = []
    last_obj = None

    # Используем client.stream для асинхронного чтения по строкам
    if files:
        # multipart: тело можно передавать через data + files, но тут мы передаём payload как json string field
        # Если Ollama ожидает JSON body + files, возможно нужно адаптировать. Попробуем отправить multipart:
        data = {"payload": json.dumps(payload)}
        async with client.stream("POST", OLLAMA_URL, data=data, files=files, timeout=timeout) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                line = line.strip()
                try:
                    obj = json.loads(line)
                except Exception:
                    # если это не JSON — пропускаем
                    continue
                last_obj = obj
                # извлекаем кусок текста, если есть
                c = obj.get("message", {}) and obj.get("message", {}).get("content")
                if c:
                    chunks.append(c)
                # если сервер пометил done:true — можно завершить
                if obj.get("done") is True:
                    break
    else:
        # обычный JSON POST, но сервер может стримить ответ по строкам
        async with client.stream("POST", OLLAMA_URL, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                line = line.strip()
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                last_obj = obj
                c = obj.get("message", {}) and obj.get("message", {}).get("content")
                if c:
                    chunks.append(c)
                if obj.get("done") is True:
                    break

    final_text = "".join(chunks).strip()
    return final_text, chunks, last_obj

@app.post("/vision_then_reason")
async def vision_then_reason(image: UploadFile = File(None), instruction: str = Form("")):
    """
    Синхронно: аккумулируем стрим из Ollama и возвращаем окончательный текст.
    """
    async with httpx.AsyncClient() as client:
        if image:
            img_bytes = await image.read()
            files = {"file": (image.filename, img_bytes, image.content_type)}
            payload = {
                "model": "qwen3-vl:8b",   # у тебя qwen3-vl-8b — подправь при необходимости
                "prompt": f"Describe the image and produce concise actionable summary. Instruction: {instruction}",
                "max_tokens": 512
            }
            final_text, chunks, last_obj = await stream_ollama_and_collect(client, payload, files=files)
        else:
            payload = {
                "model": "qwen3-vl:8b",
                "prompt": f"Instruction: {instruction}\nPlease answer concisely.",
                "max_tokens": 256
            }
            final_text, chunks, last_obj = await stream_ollama_and_collect(client, payload, files=None)

    return JSONResponse({"text": final_text, "chunks": chunks, "last_obj": last_obj})

@app.post("/vision_stream")
async def vision_stream(image: UploadFile = File(None), instruction: str = Form("")):
    """
    Прокси-стрим: Прямо ретрансляция чанков от Ollama клиенту в реальном времени.
    Возвращает StreamingResponse (chunked).
    """
    async def event_generator():
        async with httpx.AsyncClient() as client:
            if image:
                img_bytes = await image.read()
                files = {"file": (image.filename, img_bytes, image.content_type)}
                payload = {
                    "model": "qwen3-vl:8b",
                    "prompt": f"Describe the image and produce concise actionable summary. Instruction: {instruction}",
                    "max_tokens": 512
                }
                async with client.stream("POST", OLLAMA_URL, data={"payload": json.dumps(payload)}, files=files, timeout=120.0) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        c = obj.get("message", {}) and obj.get("message", {}).get("content")
                        if c:
                            # отправляем клиенту как простую строку (можно оформить в SSE при желании)
                            yield c
                        if obj.get("done") is True:
                            break
            else:
                payload = {
                    "model": "qwen3-vl:8b",
                    "prompt": f"Instruction: {instruction}\nPlease answer concisely.",
                    "max_tokens": 256
                }
                async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        c = obj.get("message", {}) and obj.get("message", {}).get("content")
                        if c:
                            yield c
                        if obj.get("done") is True:
                            break

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")
