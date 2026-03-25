import traceback
from typing import Any, Dict, List
from langsmith import traceable
from langchain_chroma import Chroma
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
        self.chroma_client = self.config.Chroma_client
        self.embeddings = self.config.Embeddings
        self.model = self.config.model
        self.CHROMA_COLLECTION = self.config.Chroma_collection
        self._vector_store = None
        self.MongoURI = self.config.MongoURI
        self.DB_NAME = self.config.DB_Mongo
        self.History_Collection_Name = self.config.History_Collection_Name

    @traceable(name="Initialize Vector Store")
    def initialize_vector_store(self) -> Chroma:
        """Initialize and return the vector store with Chroma server."""
        if self._vector_store is None:
            try:
                logger.info(f"Initializing vector store with collection")

                self._vector_store = Chroma(
                    client=self.chroma_client,  # Use direct client instead of property
                    collection_name=self.CHROMA_COLLECTION,
                    embedding_function=self.embeddings,
                )

                # Test the connection by getting collection info
                collection_count = self._vector_store._collection.count()
                logger.info(f"Connected to Chroma server - Collection '{self.CHROMA_COLLECTION}' has {collection_count} documents")

            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise
        return self._vector_store

    def format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents into a structured context string."""
        context_parts = []
        logger.info(f"Formatting {len(docs)} documents")
        for doc in docs:

            context_parts.append(
                f"Content: {doc.page_content}\n"
                f"Document Name: {doc.metadata}\n"
            )

        return "\n\n".join(context_parts)


    def initialize_retriever(self):
        """Initialize retriever with cosine similarity and cross-encoder reranking."""
        vector_store = self.initialize_vector_store()

        # Base retriever — fetch_k=20 gives reranker more candidates to rerank, returns top k=5
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )

        # Cross-encoder reranker — reranks the 20 candidates and returns top 5
        encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=encoder, top_n=5)

        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

    def get_mongo_session_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get chat history for a session."""
        try:
            return MongoDBChatMessageHistory(
                self.MongoURI,
                session_id,
                database_name=self.DB_NAME,
                collection_name=self.History_Collection_Name
            )
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            raise


    @traceable(name="Build RAG Chain")
    def build_rag_chain(self):
            """
            Build the RAG (Retrieval Augmented Generation) chain with validation.
            Returns a chain that validates input before proceeding with RAG operations.

            FIXED VERSION - Properly handles conversation history throughout the chain.
            ALSO FIXED: Handles Vertex AI returning lists instead of strings.
            """
            try:
                # Initialize components
                retriever = self.initialize_retriever()
                if not retriever:
                    raise ValueError("Retriever initialization failed")

                # Create standalone question chain with flexible parser
                standalone_question_prompt = self._create_standalone_question_prompt()
                question_chain = standalone_question_prompt | self.model

                retriever_chain = (
                    RunnablePassthrough.assign(
                        standalone_question=question_chain
                    )
                    .assign(
                        context=lambda x: self.format_context(
                            retriever.invoke(x["standalone_question"])
                        )
                    )
                )

                # Create RAG prompt and chain with flexible parser
                rag_prompt = self._create_rag_prompt()
                base_rag_chain = retriever_chain | rag_prompt | self.model

                def branch_based_on_validation(inputs: Dict[str, Any]) -> str:
                    """
                    Validates input and branches to either error message or RAG processing.
                    This function receives inputs WITH history already injected by RunnableWithMessageHistory.

                    CRITICAL: This function must preserve and pass through the history to base_rag_chain.
                    """

                    try:
                        # Validate input structure
                        if not isinstance(inputs, dict):
                            logger.error(f"Invalid inputs type: {type(inputs)}")
                            return "Invalid input format"

                        if "question" not in inputs:
                            logger.error("No question provided in input")
                            return "No question provided in input"

                        # Invoke RAG chain with full inputs (including history)
                        result = base_rag_chain.invoke(inputs)

                        return result

                    except Exception as e:
                        logger.error("ERROR IN VALIDATION/RAG CHAIN")
                        return "I apologize, but I'm experiencing technical difficulties. Please try again."

                # CRITICAL FIX: Wrap function in RunnableLambda so it has .with_listeners() method
                # Raw Python functions don't work with RunnableWithMessageHistory
                branch_runnable = RunnableLambda(branch_based_on_validation)

                return RunnableWithMessageHistory(
                    branch_runnable,  # Wrapped in RunnableLambda to make it a proper Runnable
                    self.get_mongo_session_history,
                    input_messages_key="question",
                    history_messages_key="history"
                )

            except Exception as e:
                logger.error(f"Failed to build RAG chain: {e}")
                logger.error(traceback.format_exc())
                raise Exception(f"Failed to build RAG chain: {str(e)}")

    @staticmethod
    def _create_rag_prompt() -> ChatPromptTemplate:
        """Create the RAG prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", """
            Your name is MACOM AI (MACOM AI Assistant), and you help employees analyze or
            learn about MACOM Company policy.

            IMPORTANT INSTRUCTIONS:
            1. Answer questions based ONLY on the following context
            2. If the context doesn't contain the answer, ask for more context
            3. Pay attention to the conversation history for context about what was previously discussed
            4. If the user asks to format previous information (e.g., "show that in a table"),
               use the chat history to understand what information they're referring to
            5. Support formatting requests like tables, bullet points, etc. when asked
            6. When creating tables, use proper markdown table formatting
            7. CRITICAL: Always complete tables fully - do not stop mid-row or mid-column
            8. For tables, ensure EVERY row is complete with all columns and closing pipes |
            9. End tables properly - complete the last row before finishing your response
            10. Do not include 'AI:' in front of your answers
            11. If formatting as a table, ensure all relevant information is included and organized clearly

            Context:
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

    @staticmethod
    def _create_standalone_question_prompt() -> ChatPromptTemplate:
        """Create the standalone question prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", """
            Given a chat history and a follow-up question, rephrase the follow-up
            question to be a standalone question. Do NOT answer the question, just
            reformulate it if needed, otherwise return it as is. Only return the
            final standalone question.

            If the user asks to format previous information differently (e.g., "show that in a table"),
            you should rephrase it to reference what specific information should be shown in a table.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])