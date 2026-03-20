from guardrails_check import check_guardrails
import asyncio

from src.init.singleton import Init

class Chatbot_Pipeline:
    pipeline = Init()

    def main_chatbot(input_text):
        guard_result = asyncio.run(check_guardrails(input_text))

        if guard_result == True:
            response = Chatbot_Pipeline.pipeline.run(input_text)
            return response

        if guard_result[]

