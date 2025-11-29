# app/pipeline.py
import json
from typing import Dict, Any, Optional, List
from .config import load_prompts
from .ollama_client import OllamaClient

class Pipeline:
    def __init__(self, cfg, prompts: Dict[str,Any], client: OllamaClient):
        self.cfg = cfg
        self.prompts = prompts
        self.client = client
        # load commonly used items
        self.system_role = prompts["system"]["general"]["role"]
        self.disclaimer = prompts["system"]["general"]["disclaimer"]

    def _render(self, template: str, **kwargs) -> str:
        return template.format(system_role=self.system_role, disclaimer=self.disclaimer, **kwargs)

    async def ask_questions(self, image_bytes: bytes, filename: str, stream: bool=False, max_tokens: Optional[int]=None):
        tmpl = self.prompts["templates"]["ask_questions"]
        system_text = self._render(tmpl["system"])
        user_text = self._render(tmpl["user"], image_description="(image attached)")

        prompt = f"System: {system_text}\n\nUser: {user_text}"
        payload = {
            "model": self.cfg.ollama.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.cfg.ollama.defaults.get("max_tokens", 256),
            "stream": stream
        }
        files = {"file": (filename, image_bytes, "image/jpeg")}
        if stream:
            final, chunks, last = await self.client.call_stream(payload, files=files)
            # parse lines into questions
            qlist = []
            for i, line in enumerate(final.splitlines()):
                if line.strip():
                    qlist.append({"id": f"q{i+1}", "text": line.strip()})
                    if len(qlist) >= 3:
                        break
            return {"stage":"asking", "questions": qlist, "chunks": chunks, "ollama_last": last}
        else:
            resp = await self.client.call_nostream(payload, files=files)
            # extract human-readable text
            text_out = self._extract_text(resp)
            qlist = []
            for i, line in enumerate(text_out.splitlines()):
                if line.strip():
                    qlist.append({"id": f"q{i+1}", "text": line.strip()})
                    if len(qlist) >= 3:
                        break
            return {"stage":"asking", "questions": qlist, "ollama_raw": resp, "image_description": text_out}

    async def finalize(self, image_description: str, answers: Dict[str,Any], stream: bool=False, max_tokens: Optional[int]=None):
        tmpl = self.prompts["templates"]["final_diagnosis"]
        system_text = self._render(tmpl["system"])
        user_text = self._render(tmpl["user"], image_description=image_description, answers=json.dumps(answers))

        prompt = f"System: {system_text}\n\nUser: {user_text}"
        payload = {
            "model": self.cfg.ollama.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.cfg.ollama.defaults.get("max_tokens", 512),
            "stream": stream
        }

        if stream:
            final, chunks, last = await self.client.call_stream(payload)
            parsed = None
            try:
                parsed = json.loads(final)
            except Exception:
                parsed = None
            return {"stage":"final", "final_text": final, "parsed": parsed, "chunks": chunks, "ollama_last": last}
        else:
            resp = await self.client.call_nostream(payload)
            text_out = self._extract_text(resp)
            parsed = None
            try:
                parsed = json.loads(text_out)
            except Exception:
                parsed = None
            return {"stage":"final", "final_text": text_out, "parsed": parsed, "ollama_raw": resp}

    def _extract_text(self, resp: Dict[str,Any]) -> str:
        for k in ("output","text","result"):
            if k in resp:
                return resp[k]
        # fallback to choices style
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(resp)
