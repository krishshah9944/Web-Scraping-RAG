# app.py
import os
import shutil
import asyncio
import streamlit as st
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure credentials
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize models
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Async crawling function
async def async_crawl(url, sitemap_url=None):
    """Crawl website using crawl4ai's async API"""
    browser_config = BrowserConfig(
        headless=True,  # Run in background
        proxy=None,
        
    )
    
    run_config = CrawlerRunConfig(
        
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        return result.markdown or result.raw_html

def get_db_stats():
    """Get database statistics"""
    stats = {"document_count": 0, "sources": set(), "avg_length": 0}
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("langchain")
        if collection:
            stats["document_count"] = collection.count()
            docs = collection.get(include=["metadatas", "documents"])
            total_len = sum(len(d) for d in docs["documents"])
            stats["sources"] = {m["source"] for m in docs["metadatas"] if m.get("source")}
            stats["sources_count"] = len(stats["sources"])
            stats["avg_length"] = total_len / stats["document_count"] if stats["document_count"] > 0 else 0
    except Exception:
        pass
    return stats

def clean_database():
    """Clean ChromaDB database"""
    try:
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.session_state.messages = []
        st.success("Database cleaned successfully!")
    except Exception as e:
        st.error(f"Error cleaning database: {str(e)}")

def process_content(content, source_url):
    """Process crawled content into documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.create_documents(
        [content],
        metadatas=[{"source": source_url} for _ in [content]]
    )
    return docs

# Streamlit UI
st.title("🌐 AI Website Analyzer")

# Dashboard
st.subheader("Database Status")
stats = get_db_stats()
cols = st.columns(4)
cols[0].metric("Documents", stats["document_count"] or "-")
cols[1].metric("Sources", stats.get("sources_count", 0) or "-")
cols[2].metric("Avg Length", f"{stats['avg_length']:.0f} chars" if stats['avg_length'] > 0 else "-")
cols[3].button("🧹 Clean DB", on_click=clean_database)

# Sidebar controls
with st.sidebar:
    st.header("Website Setup")
    url = st.text_input("Enter URL*", placeholder="https://example.com")
    sitemap = st.text_input("Sitemap URL", placeholder="https://example.com/sitemap.xml")
    
    if st.button("🚀 Crawl & Process"):
        if url:
            async def process_website():
                with st.spinner("Crawling website..."):
                    content = await async_crawl(url, sitemap)
                    if content:
                        with st.spinner("Processing content"):
                            docs = process_content(content, url)
                            vector_db = Chroma.from_documents(
                                documents=docs,
                                embedding=embeddings,
                                persist_directory="./chroma_db"
                            )
                            vector_db.persist()
                            st.success(f"Processed {len(docs)} chunks!")
            asyncio.run(process_website())
        else:
            st.warning("Please enter a website URL")

# Chat interface
st.subheader("Chat with Content")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about the website"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    try:
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        if vector_db._collection.count() == 0:
            st.warning("Database empty! Crawl a website first.")
            st.stop()
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        with st.spinner("Analyzing..."):
            response = qa_chain.invoke({"query": prompt})
            
        with st.chat_message("assistant"):
            st.write(response["result"])
            with st.expander("Sources"):
                sources = {doc.metadata["source"] for doc in response["source_documents"]}
                st.write("\n".join(f"- {s}" for s in sources))
                
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
        
    except Exception as e:
        st.error(f"Error: {str(e)}")