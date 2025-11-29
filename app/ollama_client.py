# app/ollama_client.py
import json
import httpx
from typing import Dict, Any, Optional, Tuple, List

class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 120):
        self.base_url = base_url
        self.timeout = timeout_seconds

    async def call_stream(self, payload: Dict[str, Any]) -> Tuple[str, List[str], Optional[Dict[str,Any]]]:
        payload = payload.copy()
        payload["stream"] = True
        chunks: List[str] = []
        last_obj = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.base_url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    last_obj = obj
                    msg = obj.get("message")
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if c:
                            chunks.append(c)
                    if obj.get("done") is True:
                        break
        final_text = "".join(chunks).strip()
        return final_text, chunks, last_obj

    async def call_nostream(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = payload.copy()
        p["stream"] = False
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=p)
            resp.raise_for_status()
            return resp.json()
