import os
from functools import lru_cache
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from nemoguardrails import RailsConfig, LLMRails
from guardrails_action import run_rag_pipeline
from src.init.singleton import Init
from src.logging import logger

_GUARDRAILS_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

@lru_cache(maxsize=1)
def get_rails() -> LLMRails:
    # instance = Init()
    with open(os.path.join(_GUARDRAILS_DIR, "config.yml"), encoding="utf-8") as f:
        yaml_content = f.read()
    with open(os.path.join(_GUARDRAILS_DIR, "greetings.co"), encoding="utf-8") as f:
        colang_content = f.read()

    config = RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content,
    )
    logger.info("NeMo Guardrails loaded successfully.")

    # LLMRails cached here — embedding index built ONCE, reused every call
    rails = LLMRails(config, verbose=True)
    rails.register_action(run_rag_pipeline, name="run_rag_pipeline")
    logger.info("NeMo Guardrails loaded successfully.")
    return rails   # ← return LLMRails, not config
