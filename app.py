import re
from typing import Annotated, Dict
import httpx
from pymongo import MongoClient
import uvicorn
import urllib.parse
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Query, Request, Response # Corrected Import
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.cors import CORSMiddleware

# Project imports
from chat import Chatbot_Pipeline
from src.pipeline.Clear_history import ClearHistory
from src.pipeline.Login import Login
from src.pipeline.Text_To_Speach import TextToSpeach
from src.pipeline.history import ChatHistoryResponse, ChatMessage, HistoryPage
from src.components.token import Token # Ensure this is imported
from src.entity import ChatHistoryClear, ChatRequest2, EncryptedLoginData, TextToSpeechRequest
from src.utils.common import decrypt_credentials
from src.logging import logger
from src.utils.security import (
    SecurityHeadersMiddleware,
    RestrictSwaggerMiddleware,
    BlockReDocMiddleware,
    check_no_query_params
)

routes = APIRouter()

@routes.post("/login")
async def login(request: Request, data: EncryptedLoginData = Body(...)):
    try:
        # 1. Security Check
        check_no_query_params(request)

        # 2. Decryption Logic
        userName = urllib.parse.unquote(data.userName)
        password = data.password

        decrypted = decrypt_credentials({
            "username": userName,
            "password": password
        })

        if not decrypted.get('username') or not decrypted.get('password'):
            raise HTTPException(status_code=400, detail="Credentials required")

        # 3. Initialize Login (Inject Token dependency if needed)
        # Assuming Login class now handles its own singleton config internally
        login_service = Login(token=Token())
        result = login_service.login_user(decrypted['username'], decrypted['password'])

        # 4. Handle Response
        if result.get("status") == "success":
            user_status = result.get("user_Status")
            access_token = result.get("access_token")

            # Consolidated active/inactive logic
            response = JSONResponse(content={
                'status': 'success',
                'user_Status': user_status
            })
            # Important: Set the header
            response.headers['Authorization'] = f"Bearer {access_token}"
            return response

        else:
            # Return specific error from login service (e.g., 401 Unauthorized)
            return JSONResponse(result, status_code=401)

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        # Catch-all for unexpected issues
        raise HTTPException(status_code=500, detail=str(e))


@routes.post("/chat2")
async def chat(request: Request, data: ChatRequest2 = Body(...)):
    try:
        check_no_query_params(request)
        authorization_header = request.headers.get("Authorization")
        if authorization_header and authorization_header.startswith("Bearer "):
            access_token = authorization_header[len("Bearer "):].strip()
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
        user_input = urllib.parse.unquote(data.input)
        lang = urllib.parse.unquote(data.lang)
        sanitized_input = re.sub(r'[<>{}[\]\\|]', '', user_input)
        bot = Chatbot_Pipeline()
        result = await bot.main_chatbot(access_token, sanitized_input, lang)
        logger.info("Chat request completed")
        return JSONResponse(content=result)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@routes.post("/chat_history/{user_id}",response_model=ChatHistoryResponse)
async def get_chat_history(
    user_id: str,
    request : Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
    mongo_client: MongoClient = Depends(HistoryPage.get_mongo_client),
) -> ChatHistoryResponse:
    authorization_header = request.headers.get("Authorization")

    if authorization_header and authorization_header.startswith("Bearer "):
        access_token = authorization_header[len("Bearer "):].strip()
    else:
        raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")


    try:
        chat=Token()
        tok_data=chat.validate_access_token(access_token)
        if not tok_data:
            raise HTTPException(status_code=401, detail="Invalid or expired access token")

        new_access_token=chat.create_update_token(tok_data)
        history = HistoryPage.get_chat_history_from_mongodb(
            user_id=user_id,
            mongo_client=mongo_client,
            limit=limit,
            offset=offset,
        )

        chat_messages = []
        if history:
            for message_doc in history:
                try:
                    query_content = message_doc.get("query")
                    response_content = message_doc.get("query_response")
                    timestamp_str = message_doc.get("timestamp")

                    created_at = HistoryPage.parse_timestamp_string(timestamp_str)

                    if query_content:
                        user_chat_message = ChatMessage(
                            role="user",
                            content=query_content,
                            created_at=created_at,
                        )
                        chat_messages.append(user_chat_message)

                    if response_content:
                        ai_chat_message = ChatMessage(
                            role="Ai",
                            content=response_content,
                            created_at=created_at,
                        )
                        chat_messages.append(ai_chat_message)

                except Exception as e:
                    logger.error(f"Error processing message document {message_doc.get('_id')}: {e}", exc_info=True)
        else:
            logger.info(f"No chat history found for user: {user_id}. Returning empty list.")

        chat_response = ChatHistoryResponse(
            user_id=user_id, messages=chat_messages
        )
        return Response(
                content=chat_response.model_dump_json(),
                headers={"Authorization": f"Bearer {new_access_token}"},
                media_type="application/json"
            )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in get_chat_history endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chat history due to an internal server error."
        )


@routes.post("/clear_history")
async def clear_chat_history(request: Request):
    # 1. Security check
    check_no_query_params(request)

    # 2. Extract Token from Header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Header")

    token = auth_header.split(" ")[1]

    # 3. Process
    handler = ClearHistory()
    return handler.clear_history(token)


@routes.post("/proxy-verify-otp")
async def proxy_verify_otp(request: Request, otp_data: Dict[str, str] = Body(...)):
    """
    Proxies the OTP verification request to the external LDAP service.
    """
    try:
        otp = otp_data.get("otp")
        if not otp:
            raise HTTPException(status_code=400, detail="OTP is required")

        external_url = "https://docker.mactech.net.in:5013/ldap-service/verifyLoginOtp"

        async with httpx.AsyncClient() as client:
            external_response = await client.post(
                external_url,
                json={"otp": otp},
                headers={"Content-Type": "application/json"}
            )

            external_response.raise_for_status()

            return JSONResponse(content=external_response.json(), status_code=external_response.status_code)

    except httpx.HTTPStatusError as e:
        logger.error(f"External OTP service responded with error: {e.response.status_code} - {e.response.text}")
        return JSONResponse(
            content={"detail": f"External OTP service error: {e.response.status_code} - {e.response.text}"},
            status_code=e.response.status_code
        )
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to external OTP service: {e}")
        raise HTTPException(status_code=503, detail=f"Cannot connect to external OTP service: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in proxy-verify-otp: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during OTP proxy.")





@routes.post("/text-to-speech")
async def text_to_speech(request: Request, data: TextToSpeechRequest = Body(...)):
    try:
        check_no_query_params(request)
        authorization_header = request.headers.get("Authorization")
        if authorization_header and authorization_header.startswith("Bearer "):
            access_token = authorization_header[len("Bearer "):].strip()
        else:
            raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")
        text_to_speech = TextToSpeach()
        response=text_to_speech.Text_to_speech_process(data,access_token)
        return response

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
















def init_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)

    # Middleware setup
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["POST"],
        allow_headers=["*"], # Simplified for development
        expose_headers=["Authorization"], # Essential so frontend can see the token
    )

    app.include_router(routes)
    return app

if __name__ == "__main__":
    app = init_app()
    uvicorn.run(app, host="0.0.0.0", port=5050)