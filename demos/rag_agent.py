# Retrieval augmented generation (RAG)
# https://python.langchain.com/docs/concepts/rag/

from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, LLMTextFrame, TranscriptionFrame, TTSSpeakFrame
from loguru import logger

load_dotenv()

class RagAgentProcessor(FrameProcessor):
    def __init__(
        self, 
        topic, 
        llm_model=None, 
        llm_model_provider=None, 
        embeddings_model=None, 
        webpage_documents=None,
        additional_context=None,
        name=None
    ):
        super().__init__(name=name or f"RAG-{topic}")
        
        self.topic = topic
        self.system_prompt = """You are an expert in the topic of {topic} and you are here to answer any questions you have regarding the topic.
Use the following pieces of retrieved context, which provide information about {topic}, to answer the questions.
Use three sentences maximum and keep the answer concise.
Context: {additional_context}
{context}:"""
        self.greeting = f"Hello! I am an expert in {topic}. Ask me any questions you have regarding the topic."
        self.llm_model = llm_model
        self.llm_model_provider = llm_model_provider
        self.embeddings_model = embeddings_model
        self.webpage_documents = webpage_documents
        self.additional_context = additional_context
        self.vector_store = None
        self.llm = None
        self.memory = None
        self.agent_executor = None
        self.config = {"configurable": {"thread_id": "def234"}}
        
        # Initialize components
        self._setup_retrieval()
        self._setup_llm()
        self._setup_memory()
        self._setup_agent()
        
        logger.info(f"RAG Agent initialized for topic: {topic}")

    def _setup_retrieval(self):
        """Set up the retrieval components."""
        embeddings = VertexAIEmbeddings(model=self.embeddings_model)
        self.vector_store = InMemoryVectorStore(embeddings)
        
        if self.webpage_documents:
            loader = WebBaseLoader(web_paths=self.webpage_documents)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(docs)
            _ = self.vector_store.add_documents(documents=all_splits)
            logger.info(f"Loaded {len(all_splits)} document chunks into vector store")
    
    def generate_retrieve_tool(self):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        return retrieve
    
    def _setup_llm(self):
        """Set up the language model."""
        self.llm = init_chat_model(self.llm_model, model_provider=self.llm_model_provider)
    
    def _setup_memory(self):
        """Set up the memory component."""
        self.memory = MemorySaver()
    
    def _setup_agent(self):
        """Set up the agent executor."""
        prompt = self.system_prompt.format(topic=self.topic, context="{context}", additional_context=self.additional_context)
        logger.info(f"Setting up agent executor with prompt: {prompt}")
        self.agent_executor = create_react_agent(
            model=self.llm, 
            tools=[self.generate_retrieve_tool()], 
            prompt=prompt,
            checkpointer=self.memory
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction) # Ensure base processing happens

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame) and frame.text and frame.text.strip():
            response = await self._rag_query(frame.text)
            await self.push_frame(TTSSpeakFrame(response))
            # RAG Agent
            
        else:
            # Let other frames pass through
            await self.push_frame(frame, direction)

    async def _rag_query(self, query: str) -> str:
        """Process a query using the RAG agent and return the response."""
        # Start metrics if enabled
        # await self.start_processing_metrics()
        
        # Use the agent to process the query
        response_stream = self.agent_executor.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            config=self.config
        )
        
        # Collect the response
        full_response = ""
        for chunk in response_stream:
            if "messages" in chunk and chunk["messages"]:
                message = chunk["messages"][-1]
                if isinstance(message, AIMessage) or (isinstance(message, dict) and message.get("role") == "assistant"):
                    full_response = message["content"] if isinstance(message, dict) else message.content
        
        # Stop metrics if enabled
        # await self.stop_processing_metrics()
        
        return full_response
    
    def can_generate_metrics(self) -> bool:
        """Indicate that this processor can generate metrics."""
        return True
    
    async def send_greeting(self):
        """Send the greeting as an LLMTextFrame."""
        greeting_frame = LLMTextFrame(self.greeting)
        await self.push_frame(greeting_frame)
