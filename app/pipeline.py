# app/pipeline.py
import json
from typing import Dict, Any, Optional, List, Tuple
from .config import load_prompts
from .ollama_client import OllamaClient
import base64


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

    def _split_description_and_questions(self, text: str) -> Tuple[str, List[str]]:
        """
        Делит ответ модели на:
        - description: часть с visual description
        - questions: строки-вопросы (те, что заканчиваются '?')
        """
        # нормализуем переносы
        t = text.replace("\r\n", "\n")

        # поищем "Step 2" или "Clarifying questions"
        lower = t.lower()
        idx_step2 = lower.find("step 2")
        if idx_step2 != -1:
            desc_part = t[:idx_step2].strip()
            rest = t[idx_step2:]
        else:
            # если нет явного разделения — считаем всё описанием,
            # а вопросы выделим по "?" в конце строк
            desc_part = t
            rest = t

        # из оставшейся части вытаскиваем строки с вопросами
        questions: List[str] = []
        for line in rest.splitlines():
            line = line.strip()
            if not line:
                continue
            # берём только строки, которые выглядят как вопросы
            if "?" in line and line.strip().endswith("?"):
                questions.append(line)

        # если вообще не нашли вопросов — fallback: ничего не меняем
        return desc_part, questions

    async def ask_questions(self, image_bytes: bytes, filename: str,
                            stream: bool=False, max_tokens: Optional[int]=None):
        tmpl = self.prompts["templates"]["ask_questions"]
        system_text = self._render(tmpl["system"])
        user_text = self._render(tmpl["user"], image_description="(image attached)")

        prompt = f"System: {system_text}\n\nUser: {user_text}"

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.cfg.ollama.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.cfg.ollama.defaults.get("max_tokens", 256),
            "images": [img_b64],
            "stream": stream,
        }

        if stream:
            final, chunks, last = await self.client.call_stream(payload)
            description, questions = self._split_description_and_questions(final)
            qlist = [{"id": f"q{i+1}", "text": q} for i, q in enumerate(questions)]
            return {
                "stage": "asking",
                "image_description": description,
                "questions": qlist,
                "chunks": chunks,
                "ollama_last": last,
            }
        else:
            resp = await self.client.call_nostream(payload)
            text_out = self._extract_text(resp)
            description, questions = self._split_description_and_questions(text_out)
            qlist = [{"id": f"q{i+1}", "text": q} for i, q in enumerate(questions)]
            return {
                "stage": "asking",
                "image_description": description,
                "questions": qlist,
                "ollama_raw": resp,
            }

    async def finalize(
        self,
        image_description: str,
        qa_pairs: List[Dict[str, str]],
        stream: bool = False,
        max_tokens: Optional[int] = None,
    ):
        tmpl = self.prompts["templates"]["final_diagnosis"]
        system_text = self._render(tmpl["system"])

        # qc_json = список объектов {question, answer}, чтобы модель видела семантику
        answers_json = json.dumps(qa_pairs, ensure_ascii=False)

        user_text = self._render(
            tmpl["user"],
            image_description=image_description,
            answers=answers_json,
        )

        prompt = f"System: {system_text}\n\nUser: {user_text}"
        payload = {
            "model": self.cfg.ollama.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.cfg.ollama.defaults.get("max_tokens", 512),
            "stream": stream,
        }

        if stream:
            final, chunks, last = await self.client.call_stream(payload)
            parsed = None
            try:
                parsed = json.loads(final)
            except Exception:
                parsed = None
            return {
                "stage": "final",
                "final_text": final,
                "parsed": parsed,
                "chunks": chunks,
                "ollama_last": last,
            }
        else:
            resp = await self.client.call_nostream(payload)
            text_out = self._extract_text(resp)
            parsed = None
            try:
                parsed = json.loads(text_out)
            except Exception:
                parsed = None
            return {
                "stage": "final",
                "final_text": text_out,
                "parsed": parsed,
                "ollama_raw": resp,
            }
    def _extract_text(self, resp: Dict[str, Any]) -> str:
        # Qwen через Ollama: основной текст лежит в поле "response"
        if "response" in resp and isinstance(resp["response"], str):
            return resp["response"]

        # запасные варианты (на будущее / другие модели)
        for k in ("output", "text", "result"):
            if k in resp and isinstance(resp[k], str):
                return resp[k]

        # OpenAI-like формат
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            # fallback - отладка: вернуть весь json
            return json.dumps(resp)

