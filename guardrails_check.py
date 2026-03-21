import logging
from nemoguardrails import RailsConfig, LLMRails
from src.logging import logger

# Load once at module level — avoids reloading on every request
_rails = None

def get_rails() -> LLMRails:
    global _rails
    if _rails is None:
        try:
            config = RailsConfig.from_path("guardrails/")
            _rails = LLMRails(config)
            logger.info("NeMo Guardrails loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NeMo Guardrails: {e}")
            raise
    return _rails


async def check_guardrails(user_input: str) -> dict:
    """
    Returns:
        {
          "blocked": True/False,
          "reply": "bot reply if blocked or greeting, else None"
        }
    """
    try:
        rails = get_rails()
        messages = [{"role": "user", "content": user_input}]
        response = await rails.generate_async(messages=messages)

        bot_reply = response.get("content", "").strip()

        # If guardrails produced a reply (greeting or block), return it
        if bot_reply:
            return {"blocked": True, "reply": bot_reply}

        # No guardrail triggered — let the normal pipeline handle it
        return {"blocked": False, "reply": None}

    except Exception as e:
        logger.error(f"Guardrails check failed: {e}")
        # Fail open — don't break the app if guardrails crash
        return {"blocked": False, "reply": None}