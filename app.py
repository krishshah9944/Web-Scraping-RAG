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
import requests
from xml.etree import ElementTree
from urllib.parse import urlparse,urljoin 


load_dotenv()


google_api_key = st.secrets["GOOGLE_API_KEY"]
groq_api_key = os.getenv("GROQ_API_KEY")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")


def get_urls_from_sitemap(sitemap_url: str) -> list:
    """Fetch and parse URLs from a sitemap."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [loc.text for loc in root.findall(".//ns:loc", namespace)]
    except Exception as e:
        st.error(f"Error fetching sitemap: {str(e)}")
        return []


async def async_crawl(urls: list):
    """Crawl multiple URLs using crawl4ai's async API with sitemap discovery."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    
    crawl_config = CrawlerRunConfig(
        word_count_threshold=10,
        exclude_external_links=True,
        remove_overlay_elements=True,
        process_iframes=True,
        cache_mode="bypass",
       
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        results = []
        visited_urls = set()
        queue = urls.copy()

        while queue:
            url = queue.pop(0)
            if url in visited_urls:
                continue

            visited_urls.add(url)
            result = await crawler.arun(url=url, config=crawl_config)
            
            if result.success:
                results.append({
                    "url": url,
                    "content": result.markdown
                })

                
                sitemap_url = urljoin(url, "/sitemap.xml")
                if sitemap_url not in visited_urls:
                    sitemap_urls = get_urls_from_sitemap(sitemap_url)
                    if sitemap_urls:
                        st.info(f"Found sitemap at {sitemap_url} with {len(sitemap_urls)} URLs")
                        queue.extend(sitemap_urls)
                        visited_urls.add(sitemap_url)

        return results
    finally:
        await crawler.close()


from langchain.schema import Document

def process_content(content: list):
    """Process crawled content into LangChain Document objects."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = []
    for page in content:
        chunks = text_splitter.split_text(page["content"])
        for chunk in chunks:
            parsed_url = urlparse(page["url"])
            
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": page["url"],
                    "domain": parsed_url.netloc
                }
            ))
    return documents


st.title("🌐 AI Website Analyzer")


st.subheader("Database Status")
def get_db_stats():
    """Get database statistics."""
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

stats = get_db_stats()
cols = st.columns(4)
cols[0].metric("Documents", stats["document_count"] or "-")
cols[1].metric("Sources", stats.get("sources_count", 0) or "-")
cols[2].metric("Avg Length", f"{stats['avg_length']:.0f} chars" if stats['avg_length'] > 0 else "-")
cols[3].button("🧹 Clean DB", on_click=lambda: clean_database())

def clean_database():
    """Clean ChromaDB database."""
    try:
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.session_state.messages = []
        st.success("Database cleaned successfully!")
    except Exception as e:
        st.error(f"Error cleaning database: {str(e)}")


with st.sidebar:
    st.header("Website Setup")
    url = st.text_input("Enter URL*", placeholder="https://example.com")
    sitemap = st.text_input("Sitemap URL", placeholder="https://example.com/sitemap.xml")
    
    if st.button("🚀 Crawl & Process"):
        if url:
            async def process_website():
                with st.spinner("Crawling website..."):
                    urls = [url]
                    if sitemap:
                        sitemap_urls = get_urls_from_sitemap(sitemap)
                        if sitemap_urls:
                            urls = sitemap_urls
                    
                    content = await async_crawl(urls)
                    if content:
                        with st.spinner("Processing content"):
                            processed_content = process_content(content)
                            vector_db = Chroma.from_documents(
                                documents=processed_content,
                                embedding=embeddings,
                                persist_directory="./chroma_db"
                            )
                            vector_db.persist()
                            st.success(f"Processed {len(processed_content)} chunks!")
            asyncio.run(process_website())
        else:
            st.warning("Please enter a website URL")


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


