from init.singleton import Init
from src.config.configuration import ConfigurationManager
from src.components.token import Token
from src.logging import logger
from pymongo import MongoClient
from fastapi import HTTPException
from fastapi.responses import JSONResponse
class ClearHistory:
    def __init__(self):
        try:
            self.config = Init()
            self.token = Token()
            self.history_collection = self.config.DB[self.config.History_Collection_Name]
        except Exception as e:
            logger.error(f"Error initializing Login class: {e}")
            raise

    def clear_history_process(self,session_id: str):
        try:
            # Count the number of messages for the given session
            message_count = self.history_collection.count_documents({"SessionId": session_id})
            print(f"Total messages for session {session_id}: {message_count}")
            
            # Remove all messages for this session
            if message_count > 0:
                result = self.history_collection.delete_many({"SessionId": session_id})
                print(f"Removed {result.deleted_count} messages for session {session_id}.")
            
            return {'status': 'success', 'message': 'All chat history cleared'}
        
        except Exception as e:
            print(f"Error in clearing history for session {session_id}: {e}")
            return {'status': 'error', 'message': str(e)}

    def clear_history(self, access_token: str):
            tok_data = self.token.validate_access_token(access_token)
            if not tok_data:
                raise HTTPException(status_code=401, detail="Invalid or expired Token")

            session_id = tok_data.get("session_id")
            # Perform the deletion
            process_result = self.clear_history_process(session_id)
            
            if process_result['status'] == 'error':
                raise HTTPException(status_code=500, detail="Failed to clear history")

            # Generate a fresh token to extend the session
            new_access_token = self.token.create_update_token(tok_data)
            
            response = JSONResponse(content={'status': 'success', 'message': 'History cleared'})
            response.headers['Authorization'] = f"Bearer {new_access_token}"
            return response

        
