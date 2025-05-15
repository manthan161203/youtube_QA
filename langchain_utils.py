import os
import streamlit as st

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def process_with_langchain(transcript_text, transcript_segments=None, embed_model="huggingface"):
    """Process the transcript with LangChain."""
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Split text into chunks
    if transcript_segments and len(transcript_segments) > 0:
        # Create documents from transcript segments with timestamps
        docs = []
        for segment in transcript_segments:
            doc = Document(
                page_content=segment['text'],
                metadata={
                    "start": segment['start'],
                    "duration": segment['duration']
                }
            )
            docs.append(doc)
    else:
        # Split the full transcript
        docs = text_splitter.create_documents([transcript_text])
    
    # Initialize embeddings model
    if embed_model == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Please set your OpenAI API key in the sidebar to use this embedding model.")
            return None, None
        embeddings = OpenAIEmbeddings()
    else:  # default to huggingface
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        return docs, vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None, None

def get_llm(model_name="gpt-3.5-turbo"):
    """Initialize and return LLM based on settings"""
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please set your OpenAI API key in the sidebar to use the content generation features.")
        return None
    
    return ChatOpenAI(
        temperature=0.2,
        model=model_name
    )