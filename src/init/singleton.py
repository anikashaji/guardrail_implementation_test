import threading

import chromadb
langchain_google_genai.GoogleGenerativeAI, VertexAIEmbeddings,GoogleGenerativeAIEmbeddings
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
                cls._instance = super().__new__(cls)
                #1.Initialize Config
                config_obj = ConfigurationManager()
                config = config_obj.get_base_config()
                cls._instance =config
                
                cls._instance.Chroma_client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
                cls._instance.Client = MongoClient(config.MONGODB_URI,MaxPoolSize = config.MAX_POOL_SIZE)
                cls._instance.DB =cls._instance.Client[config.DB_NAME]
                cls._instance.User_Collection = cls._instance.DB[config.collection_user]
                cls._instance.History_Collection_Name = config.HISTORY_COLLECTION_NAME
                cls._instance.History_Collection_Logs = config.HISTORY_COLLECTION_Logs
                cls._instance.Credentials=load_gcp_credentials()
                cls._instance.Embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDD_MODEL, credentials=cls._instance.Credentials)
                cls._instance.model = GoogleGenerativeAI(model_name = config.RAG_MODEL,temperature=config.temperature,top_p =config.Top_p,top_k=config.Top_k,max_output_tokens=config.Max_output_tokens)
                
                # cls._instance.Chatbot_manager = Chatbot_Manager(config=cls._instance.config,credentials=cls._instance.credentials)
                return cls._instance