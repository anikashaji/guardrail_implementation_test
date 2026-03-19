import threading

from pymongo import MongoClient
from src.config.configuration import ConfigurationManager








class init:
    _instance = None
    _instance_lock = threading.Lock()
    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                print('Creating the object')
                cls._instance = super().__new__(cls)
                config_obj = ConfigurationManager()
                config = config_obj.get_base_config()
                client = MongoClient(config.MONGODB_URI,maxPoolSize=config.MAX_POOL_SIZE)
                db=client[config.D]
                user_collection=db[config.collection_user]
            return