import threading

import chromadb
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
from src.config.configuration import ConfigurationManager
from src.config.gcp import load_gcp_credentials


class Init:
    _instance = None
    _instance_lock = threading.Lock()
    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                print('Creating the object')
                instance = super().__new__(cls)
                #1.Initialize Config
                config_obj = ConfigurationManager()
                config = config_obj.get_base_config()

                instance.Chroma_client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
                instance.Client = MongoClient(config.MONGODB_URI,MaxPoolSize = config.MAX_POOL_SIZE)
                instance.DB =instance.Client[config.DB_NAME]
                instance.User_Collection = instance.DB[config.collection_user]
                instance.History_Collection_Name = config.HISTORY_COLLECTION_NAME
                instance.History_Collection_Logs = config.HISTORY_COLLECTION_Logs
                instance.Credentials = load_gcp_credentials()
                instance.ALGORITHM = config.ALGORITHM
                instance.ACCESS_TOKEN_EXPIRE_MINUTES = config.ACCESS_TOKEN_EXPIRE_MINUTES
                instance.SECRET_KEY = config.SECRET_KEY
                instance.API_KEY = config.API_KEY
                instance.MongoURI = config.MONGODB_URI
                instance.DB_Mongo = config.DB_NAME
                instance.Chroma_collection = config.CHROMA_COLLECTION




                instance.Embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDD_MODEL, credentials=instance.Credentials)
                instance.model = GoogleGenerativeAI(model = config.RAG_MODEL,temperature=config.TEMPERATURE,top_p =config.TOP_P,top_k=config.TOP_K,max_output_tokens=config.MAX_OUTPUT_TOKENS,credentials=instance.Credentials)

                cls._instance = instance
             # cls._instance.Chatbot_manager = Chatbot_Manager(config=cls._instance.config,credentials=cls._instance.credentials)
        return cls._instance