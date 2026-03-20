import os
from fastapi import FastAPI, HTTPException, Depends, Query, APIRouter
from typing import Any, List, Optional, Dict, Annotated
from pydantic import BaseModel
from pymongo import MongoClient
from src.logging import logger
from datetime import datetime

from init.singleton import Init

# Define a Pydantic model for the chat message.
class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: Optional[datetime] # Changed to datetime for proper parsing


# Define a Pydantic model for the history response
class ChatHistoryResponse(BaseModel):
    user_id: str # Changed from session_id to user_id
    messages: List[ChatMessage]


# Dependency to get the MongoDB client.
class HistoryPage:
    def __init__(self):
        self.config = Init()
        self.MONGODB_URI = self.config.MongoURI
        self.DB_NAME = self.config.DB_Mongo
        self.HISTORY_COLLECTION_NAME = self.config.History_Collection_Logs
        self.client = None

    def get_mongo_client(self) -> MongoClient:
        if self.client is None:
            try:
                self.client = self.config.Client
                logger.info("Successfully connected to MongoDB")
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {e}")
                raise  # Re-raise the exception to be caught by FastAPI
        return self.client

    def close_mongo_client(self):
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Disconnected from MongoDB")

    def parse_timestamp_string(timestamp_str: str) -> datetime | None:
    #"""
    #Parses a timestamp string like '16/05/2025 11:37 AM IST' into a datetime object.
    #"""
        try:
            # Remove ' IST' and parse
            timestamp_str_no_tz = timestamp_str.replace(" IST", "").strip()
            # Define the format string for parsing
            # %d: day, %m: month, %Y: year, %I: hour (12-hour), %M: minute, %p: AM/PM
            return datetime.strptime(timestamp_str_no_tz, "%d/%m/%Y %I:%M %p")
        except ValueError:
            logger.warning(f"Could not parse timestamp string: '{timestamp_str}'")
            return None

    # Function to get chat history from MongoDB
    def get_chat_history_from_mongodb(
        self,
        user_id: str, # Changed from session_id to user_id
        mongo_client: MongoClient,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str,Any]]:
        """
        Fetches a paginated chat history for a given user ID from MongoDB,
        using the new document structure.
        """
        try:
            db = mongo_client[self.DB_NAME]
            history_collection = db[self.HISTORY_COLLECTION_NAME]

            # --- CRUCIAL CHANGE: Query by 'user' field ---
            history_cursor = (
                history_collection.find({"user": user_id}) # <-- Corrected field name to 'user'
                .sort("timestamp",1) # Sort by timestamp ascending to get chronological order
                .skip(offset)
                .limit(limit)
            )
            # --- END OF CHANGE ---

            history = list(history_cursor)
            
            # Optional: Add logging to confirm what was found
            if not history:
                logger.info(f"No history found for user: {user_id} in collection: {self.config.HISTORY_COLLECTION_NAME}")
            else:
                logger.debug(f"Fetched {len(history)} records for user: {user_id}")
                # Consider logging first few characters of a record for verification during debug
                # logger.debug(f"First history record: {history[0]}")

            return history
        except Exception as e:
            logger.error(f" No History", exc_info=True)
            raise  # Re-raise to let FastAPI handle the error (as per your original code)
