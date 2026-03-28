import threading
from guardrails_check import get_rails
from src.components.token import Token
from src.pipeline.chatprocess import ChatProcess
from src.logging import logger


class Chatbot_Pipeline:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                try:
                    instance.Chatbot_manager = ChatProcess()
                    instance.rag_chain = instance.Chatbot_manager.build_rag_chain()
                    instance.token = Token()
                    logger.info("Chatbot_Pipeline initialized successfully.")
                    cls._instance = instance
                except Exception as e:
                    logger.error(f"Chatbot_Pipeline initialization failed: {e}")
                    raise RuntimeError(f"Chatbot_Pipeline initialization failed: {e}") from e
        return cls._instance

    async def main_chatbot(self, access_token, input_text, lang):
        tok_data = self.token.validate_access_token(access_token)
        user_id = tok_data.get("sub")

        # NeMo now handles both interception AND RAG dispatch internally
        rails = get_rails()
        response = await rails.generate_async(
            messages=[{"role": "user", "content": input_text}]
        )

        if isinstance(response, dict):
            return response.get("content", "").strip()
        return str(response).strip()
