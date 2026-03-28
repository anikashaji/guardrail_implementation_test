import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from qdrant_client import QdrantClient

client = QdrantClient(url="http://10.192.5.51:6334", prefer_grpc=True)

start = time.time()
# Ping the remote server
health = client.get_collections()
print(f"Qdrant Server Ping: {time.time() - start:.2f}s")
from  src.init.singleton import Init
config = Init()
start_embed = time.time()
Embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004",credentials=config.Credentials)
test_vec = Embeddings.embed_query("test")
print(f"Vertex AI Embedding took: {time.time() - start_embed:.2f}s")


# import time
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from src.init.singleton import Init

# def test_gemini_performance():
#     # 1. Initialize from your Singleton to test actual app config
#     print("🚀 Initializing Init Singleton...")
#     config = Init()
#     model = config.model # This uses your 'GoogleGenerativeAI' instance
    
#     print(f"📊 Testing Model: {model.model}")
    
#     # 2. Simple Latency Test (Small Prompt)
#     print("\n--- Test 1: Simple Prompt ---")
#     question = "Say 'Hello, I am ready' in one sentence."
    
#     start_time = time.time()
#     try:
#         response = model.invoke(question)
#         end_time = time.time()
        
#         print(f"✅ Response: {response}")
#         print(f"⏱️ Total Latency: {end_time - start_time:.2f} seconds")
#     except Exception as e:
#         print(f"❌ Test 1 Failed: {e}")

#     # 3. Context Processing Test (Simulating RAG)
#     print("\n--- Test 2: Large Context Simulation ---")
#     context = "The company policy states that employees get 25 days of annual leave. " * 50
#     question_with_context = f"Context: {context}\n\nQuestion: How many leave days do I get?"
    
#     start_time = time.time()
#     try:
#         response = model.invoke(question_with_context)
#         end_time = time.time()
        
#         print(f"✅ Response: {response}")
#         print(f"⏱️ Processing Latency: {end_time - start_time:.2f} seconds")
#     except Exception as e:
#         print(f"❌ Test 2 Failed: {e}")

# if __name__ == "__main__":
#     test_gemini_performance()