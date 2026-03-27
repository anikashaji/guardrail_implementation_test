from nemoguardrails.actions import action
from src.logging import logger


@action(name="run_rag_pipeline")
async def run_rag_pipeline(context: dict, user_id: str = None) -> str:
    """
    NeMo will call this action when no guardrail flow intercepts the message.
    Reuses the already-built rag_chain from Chatbot_Pipeline singleton.
    """
    from chat import Chatbot_Pipeline  # deferred import to avoid circular import

    user_message = context.get("last_user_message", "")
    try:
        bot = Chatbot_Pipeline()  # returns singleton — no rebuild
        response = bot.rag_chain.invoke(
            {"question": user_message},
            {"configurable": {"session_id": user_id or "default"}}
        )
        logger.info("RAG pipeline executed via NeMo action.")
        return response
    except Exception as e:
        logger.exception(f"RAG pipeline action failed: {e}")
        return "I encountered an error processing your request."
