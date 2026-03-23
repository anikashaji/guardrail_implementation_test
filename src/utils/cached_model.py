import datetime
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logging import logger

SYSTEM_PROMPT = """You are MACOM AI Assistant, an intelligent assistant specialized in answering 
questions about MACOM Company policies and procedures.

Guidelines:
- Only answer questions related to MACOM company policies, HR procedures, and internal guidelines.
- Be concise, accurate, and professional in your responses.
- If the context provided does not contain enough information to answer the question, say so clearly.
- Do not make up information that is not present in the provided context.
- Always base your answers on the retrieved context.
- Format responses clearly using bullet points or numbered lists where appropriate.
"""


def create_cached_gemini_model(api_key: str, model_name: str, temperature: float,
                                top_p: float, top_k: int, max_output_tokens: int) -> ChatGoogleGenerativeAI:
    """
    Creates a Gemini model with cached system prompt.
    Cache TTL is 60 minutes — reused across all requests to save tokens.
    """
    try:
        client = genai.Client(api_key=api_key)

        # Create cached content with system prompt
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                system_instruction=SYSTEM_PROMPT,
                ttl=datetime.timedelta(minutes=60),
            )
        )
        logger.info(f"Gemini cache created: {cache.name} | expires: {cache.expire_time}")

        # Return LangChain-compatible model pointing to the cache
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            cached_content=cache.name,
        )

    except Exception as e:
        logger.warning(f"Failed to create Gemini cache, falling back to uncached model: {e}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
        )
