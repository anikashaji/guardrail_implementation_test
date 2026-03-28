import traceback
import time
from typing import Any, Dict, List
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from src.init.singleton import Init
from src.logging import logger

class ChatProcess:
    def __init__(self):
        self.config = Init()
        self.qdrant_client = self.config.Qdrant_client
        self.embeddings = self.config.Embeddings  
        self.model = self.config.model
        self.QDRANT_COLLECTION = self.config.Qdrant_collection 
        self._vector_store = None
        self.MongoURI = self.config.MongoURI
        self.DB_NAME = self.config.DB_Mongo
        self.History_Collection_Name = self.config.History_Collection_Name

        # FIXED: Initialize Sparse Encoder ONCE at startup to save ~8-10 seconds
        logger.info("Initializing Sparse Embedding Engine...")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Keeping the Cross-Encoder logic available but commented out as per your change
        # self.rerank_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    def initialize_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            try:
                logger.info(f"Connecting to Hybrid Qdrant collection: {self.QDRANT_COLLECTION}")
                
                self._vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.QDRANT_COLLECTION,
                    embedding=self.embeddings,
                    sparse_embedding=self.sparse_embeddings, # Using the pre-loaded engine
                    vector_name="dense",
                    sparse_vector_name="sparse", # Ensure this matches your ingestion script exactly
                    retrieval_mode=RetrievalMode.HYBRID
                )
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant: {e}")
                raise
        return self._vector_store

    def initialize_retriever(self):
        vector_store = self.initialize_vector_store()

        # FIXED: Lowered threshold slightly for Hybrid search stability
        return vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3, 
                "score_threshold": 0.7  
            }
        )

    def format_context(self, docs: List[Any]) -> str:
        if not docs:
            return "No relevant policy information found for this query."
            
        context_parts = []
        for doc in docs:
            # FIXED: 'source' is the default key for Docx2txtLoader metadata
            full_path = doc.metadata.get('source', 'Unknown Policy')
            file_name = full_path.split('\\')[-1].split('/')[-1] # Extract just the filename
            
            context_parts.append(
                f"--- Document: {file_name} ---\n"
                f"Content: {doc.page_content}\n"
            )
        return "\n\n".join(context_parts)

    def get_mongo_session_history(self, session_id: str) -> MongoDBChatMessageHistory:
        return MongoDBChatMessageHistory(
            self.MongoURI,
            session_id,
            database_name=self.DB_NAME,
            collection_name=self.History_Collection_Name
        )

    def build_rag_chain(self):
        try:
            retriever = self.initialize_retriever()

            # Chain 1: Retrieve and format context
            # Assigning context by invoking the retriever with the question
            retriever_chain = RunnablePassthrough.assign(
                context=lambda x: self.format_context(retriever.invoke(x["question"]))
            )

            # Chain 2: Prompt + Model
            rag_prompt = self._create_rag_prompt()
            
            # Combine into base RAG logic
            base_rag_chain = retriever_chain | rag_prompt | self.model

            def branch_based_on_validation(inputs: Dict[str, Any]) -> Any:
                """Handles validation and execution with history."""
                try:
                    if not inputs.get("question"):
                        return "Please provide a question so I can assist you."

                    # Invoke the actual RAG logic
                    # Vertex AI can return a list or a Message object; LangChain handles this usually
                    response = base_rag_chain.invoke(inputs)
                    
                    # Ensure we return a string (helpful for certain frontend parsers)
                    return response.content if hasattr(response, 'content') else str(response)

                except Exception as e:
                    logger.error(f"RAG Execution Error: {e}\n{traceback.format_exc()}")
                    return "I encountered an error accessing the policies. Please try again later."

            # Wrap in RunnableLambda for LangChain compatibility
            branch_runnable = RunnableLambda(branch_based_on_validation)

            return RunnableWithMessageHistory(
                branch_runnable,
                self.get_mongo_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )

        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}")
            raise

    @staticmethod
    def _create_rag_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
            Your name is MACOM AI. You help employees analyze MACOM Company policy.

            IMPORTANT: You have access to chat history below. If the user's question
            is a follow-up, infer what they mean from context — do NOT ask for clarification.
            Answer using ONLY the context provided.

            If the context doesn't contain the answer, say so clearly.
            Support formatting (tables, bullets) when requested.
            Complete all tables fully with closing pipes |.

            Context:
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])