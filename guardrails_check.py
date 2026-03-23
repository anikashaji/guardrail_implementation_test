import os
from functools import lru_cache
from nemoguardrails import RailsConfig, LLMRails
from src.logging import logger

_GUARDRAILS_DIR = os.path.join(os.path.dirname(__file__), "guardrails")


@lru_cache(maxsize=1)
def get_rails() -> LLMRails:
    with open(os.path.join(_GUARDRAILS_DIR, "config.yml")) as f:
        yaml_content = f.read()
    with open(os.path.join(_GUARDRAILS_DIR, "greetings.co")) as f:
        colang_content = f.read()
    config = RailsConfig.from_content(yaml_content=yaml_content, colang_content=colang_content)
    logger.info("NeMo Guardrails loaded successfully.")
    return LLMRails(config)


async def check_guardrails(user_input: str) -> dict:
    try:
        rails = get_rails()
        messages = [{"role": "user", "content": user_input}]
        response = await rails.generate_async(messages=messages)

        bot_reply = response.get("content", "").strip()

        if bot_reply:
            return {"blocked": True, "reply": bot_reply}

        return {"blocked": False, "reply": None}

    except Exception as e:
        logger.error(f"Guardrails check failed: {e}")
        return {"blocked": False, "reply": None}
