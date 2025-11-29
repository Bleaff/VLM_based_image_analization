# app/config.py
import os
import yaml
from pydantic import BaseModel, Field, AnyHttpUrl
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.environ.get("APP_CONFIG_PATH", "./config/config.yml")

class OllamaConfig(BaseModel):
    url: AnyHttpUrl
    default_model: str
    timeout_seconds: int
    defaults: Dict[str, Any]

class ServiceConfig(BaseModel):
    listen_host: str
    listen_port: int
    allowed_actions: list

class LoggingConfig(BaseModel):
    level: str = "INFO"

class AppConfig(BaseModel):
    ollama: OllamaConfig
    service: ServiceConfig
    logging: LoggingConfig
    prompts_file: str

def load_config(path: str = None) -> AppConfig:
    p = path or DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # allow override via env
    if "APP_OLLAMA_URL" in os.environ:
        raw.setdefault("ollama", {})["url"] = os.environ["APP_OLLAMA_URL"]
    cfg = AppConfig(**raw)
    return cfg

def load_prompts(prompts_path: str):
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
