from guardrails_check import check_guardrails
from src.components.token import Token
from src.pipeline.chatprocess import ChatProcess
from src.logging import logger


class Chatbot_Pipeline:

    def __init__(self):
        self.Chatbot_manager = ChatProcess()
        self.rag_chain = self.Chatbot_manager.build_rag_chain()
        self.token = Token()

    def _run_pipeline(self, user_input, user_id):
        return self.rag_chain.invoke(
            {"question": user_input},
            {"configurable": {"session_id": user_id}}
        )

    async def main_chatbot(self, access_token, input_text, lang):
        tok_data = self.token.validate_access_token(access_token)
        user_id = tok_data.get("sub")

        # --- Guardrails check ---
        try:
            guard_result = await check_guardrails(input_text)
            return guard_result

        except Exception as e:
            logger.warning(f"Guardrails unavailable, falling back to pipeline: {e}")

        # --- Normal pipeline ---
        try:
            response = self._run_pipeline(input_text, user_id)
            return {"response": response, "source": "pipeline"}
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
