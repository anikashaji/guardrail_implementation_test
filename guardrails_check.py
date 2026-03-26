import os
from functools import lru_cache
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from nemoguardrails import RailsConfig, LLMRails
from src.init.singleton import Init
from src.logging import logger

_GUARDRAILS_DIR = os.path.join(os.path.dirname(__file__), "guardrails")

_REFUSAL_PREFIXES = (
    "I'm sorry, I can't",
    "I'm sorry, I only",
)


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
    logger.info("NeMo Guardrails loaded successfully.")
    return rails   # ← return LLMRails, not config


async def check_guardrails(user_input: str) -> dict:
    """
    Returns:
        {"blocked": True,  "reply": str}  — dangerous/code/unrelated input, show refusal
        {"blocked": False, "reply": str}  — greeting matched by NeMo, return directly
        {"blocked": False, "reply": None} — clean policy question, forward to RAG
    """
    try:
        rails = get_rails()
        messages = [{"role": "user", "content": user_input}]
        response = await rails.generate_async(messages=messages)

        logger.debug(f"Guardrails raw response: type={type(response)!r} value={response!r}")

        # Parse NeMo response — it returns a dict for generate_async
        if isinstance(response, dict):
            bot_reply = response.get("content", "").strip()
        elif isinstance(response, list):
            assistant_msgs = [m for m in response if m.get("role") == "assistant"]
            bot_reply = assistant_msgs[-1].get("content", "").strip() if assistant_msgs else ""
        elif isinstance(response, str):
            bot_reply = response.strip()
        else:
            bot_reply = ""

        logger.info(f"Guardrails NeMo reply: {bot_reply!r}")

        # Hard block — dangerous/code/unrelated content
        if bot_reply and any(bot_reply.startswith(p) for p in _REFUSAL_PREFIXES):
            return {"blocked": True, "reply": bot_reply}

        # NeMo returned a greeting reply → surface it, skip RAG
        if bot_reply:
            # return {"blocked": False, "reply": bot_reply}
            response = JSONResponse(content={'status': 'success','answer': bot_reply}, status_code=200)
            return response


        # NeMo returned empty → clean policy question, let RAG handle it
        return {"blocked": False, "reply": None}

    except Exception as e:
        logger.exception(f"Guardrails check failed: {e}")
        return {"blocked": False, "reply": None}