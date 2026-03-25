import os
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from nemoguardrails import RailsConfig, LLMRails
from src.logging import logger
from src.init.singleton import Init

_GUARDRAILS_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

_REFUSAL_PREFIXES = ("I'm sorry, I can't",)


@lru_cache(maxsize=1)
def get_rails() -> LLMRails:
    instance = Init()
    with open(os.path.join(_GUARDRAILS_DIR, "config.yml"),encoding="utf-8") as f:
        yaml_content = f.read()
    with open(os.path.join(_GUARDRAILS_DIR, "greetings.co"),encoding="utf-8") as f:
        colang_content = f.read()
    config = RailsConfig.from_content(
        yaml_content=yaml_content,
        colang_content=colang_content,
    )
    logger.info("NeMo Guardrails loaded successfully.")

    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     max_output_tokens=256,
    #     credentials=instance.Credentials
    # )
    llm = ChatOllama(model="llama3.2:latest", base_url="http://localhost:11434")
    return LLMRails(config, llm=llm)


async def check_guardrails(user_input: str) -> dict:
    try:
        rails = get_rails()
        messages = [{"role": "user", "content": user_input}]
        response = await rails.generate_async(messages=messages)

        if isinstance(response, list):
            assistant_msgs = [m for m in response if m.get("role") == "assistant"]
            bot_reply = assistant_msgs[-1].get("content", "").strip() if assistant_msgs else ""
        elif isinstance(response, dict):
            bot_reply = response.get("content", "").strip()
        elif isinstance(response, str):
            bot_reply = response.strip()
        else:
            bot_reply = ""

        logger.info(f"Guardrails bot reply: {bot_reply!r}")

        if bot_reply and any(bot_reply.startswith(p) for p in _REFUSAL_PREFIXES):
            return {"blocked": True, "reply": bot_reply}

        if bot_reply:
            return {"blocked": False, "reply": bot_reply}

        return {"blocked": False, "reply": None}

    except Exception as e:
        logger.exception(f"Guardrails check failed: {e}")
        return {"blocked": False, "reply": None}